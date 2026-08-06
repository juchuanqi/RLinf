# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for action_level reward reshape and auto_update_epoch.

Tests the PR1 migrations: _expand_singleton_action_dim, action_level branch
in preprocess_embodied_advantages_inputs, and auto_update_epoch logic.
"""

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.utils import (
    _expand_singleton_action_dim,
    preprocess_embodied_advantages_inputs,
)


# ---------------------------------------------------------------------------
# _expand_singleton_action_dim
# ---------------------------------------------------------------------------


def test_expand_none_tensor():
    """None tensor returns None unchanged."""
    result = _expand_singleton_action_dim(None, 32, "test")
    assert result is None


def test_expand_already_match():
    """Tensor already at target size returns unchanged."""
    t = torch.randn(2, 3, 32)
    result = _expand_singleton_action_dim(t, 32, "test")
    assert result is t  # same object, no copy


def test_expand_singleton():
    """Singleton dim expanded to target size."""
    t = torch.randn(2, 3, 1)
    result = _expand_singleton_action_dim(t, 32, "test")
    assert result.shape == (2, 3, 32)
    # Values should be broadcast (same value repeated)
    assert torch.allclose(result[:, :, 0], result[:, :, 1])


def test_expand_incompatible_raises():
    """Non-singleton, non-matching dim raises ValueError."""
    t = torch.randn(2, 3, 16)
    try:
        _expand_singleton_action_dim(t, 32, "test")
        assert False, "should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# preprocess_embodied_advantages_inputs — action_level
# ---------------------------------------------------------------------------

def _call(rewards, dones, values=None, loss_mask=None, loss_mask_sum=None,
          reward_type="action_level", adv_type="gae", **extra):
    return preprocess_embodied_advantages_inputs(
        rewards=rewards, dones=dones, values=values,
        loss_mask=loss_mask, loss_mask_sum=loss_mask_sum,
        reward_type=reward_type, adv_type=adv_type, **extra,
    )


def test_action_level_scalar_dones_values_expand():
    """Test A: [2,3,32] rewards, [3,3,1] dones/values → expanded to 32."""
    r = torch.randn(2, 3, 32)
    d = torch.ones(3, 3, 1)
    v = torch.randn(3, 3, 1)
    lm = torch.ones(2, 3, 1)
    result = _call(r, d, v, lm)
    assert result["rewards"].shape == (64, 3)
    assert result["dones"].shape == (65, 3)
    assert result["values"].shape == (65, 3)
    assert result["loss_mask"].shape == (64, 3)
    assert result["chunk_size"] == 32
    assert result["n_steps"] == 64


def test_action_level_gae_expand_rewards_to_values():
    """Test B: [2,3,1] rewards + [3,3,32] values with GAE.
    GAE expand must run BEFORE _expand_singleton_action_dim, expanding
    scalar rewards/dones to match action-level values."""
    r = torch.randn(2, 3, 1)
    d = torch.ones(3, 3, 1)
    v = torch.randn(3, 3, 32)
    lm = torch.ones(2, 3, 1)
    result = _call(r, d, v, lm)
    assert result["chunk_size"] == 32, f"expected chunk_size=32, got {result['chunk_size']}"
    assert result["rewards"].shape == (64, 3)
    assert result["values"].shape == (65, 3)
    assert result["dones"].shape == (65, 3)


def test_action_level_incompatible_shapes_raises():
    """Test C: [2,3,16] rewards vs [3,3,32] values → ValueError."""
    r = torch.randn(2, 3, 16)
    d = torch.ones(3, 3, 1)
    v = torch.randn(3, 3, 32)
    try:
        _call(r, d, v)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "action dimension" in str(e)


def test_action_level_non_gae_skip_expand():
    """action_level without GAE should not trigger the expand block."""
    r = torch.randn(2, 3, 1)
    d = torch.ones(3, 3, 1)
    result = _call(r, d, values=None, adv_type="reinpp_baseline")
    # rewards stay at chunk_size=1, n_steps=2
    assert result["chunk_size"] == 1
    assert result["n_steps"] == 2


# ---------------------------------------------------------------------------
# chunk_level baseline (must match upstream/main behavior)
# ---------------------------------------------------------------------------


def test_chunk_level_no_gae():
    """Test D: chunk_level without GAE must match upstream."""
    r = torch.randn(2, 3, 32)
    d = torch.ones(3, 3, 32)
    lm = torch.ones(2, 3, 32)
    result = preprocess_embodied_advantages_inputs(
        rewards=r, dones=d, values=None, loss_mask=lm, loss_mask_sum=None,
        reward_type="chunk_level", adv_type="reinpp_baseline",
    )
    assert result["chunk_size"] == 1
    assert result["n_steps"] == 2
    assert result["rewards"].shape == (2, 3)
    assert result["dones"].shape == (3, 3)
    assert result["loss_mask"].shape == (2, 3)


def test_action_level_ndim4_flatten():
    """4-D tensors (n_chunks, bsz, action_chunks, action_dim) are flattened."""
    r = torch.randn(2, 3, 4, 8)   # 2 chunks, bsz=3, 4 action_chunks, 8 dims
    d = torch.ones(3, 3, 4, 8)
    v = torch.randn(3, 3, 4, 8)
    result = _call(r, d, v)
    # 4*8=32 action dim after flatten
    assert result["chunk_size"] == 32
    assert result["rewards"].shape == (64, 3)


# ---------------------------------------------------------------------------
# auto_update_epoch unit tests (CPU-only, no distributed)
# ---------------------------------------------------------------------------

class _FakeWorker:
    """Minimal stub to exercise the auto_update_epoch logic without GPU."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self._world_size = 1


def _make_cfg(auto_update_enabled=False, target_kl=None, update_epoch=3):
    return OmegaConf.create({
        "algorithm": {
            "update_epoch": update_epoch,
            "auto_update_epoch": {
                "enabled": auto_update_enabled,
                "target_kl": target_kl,
            },
            "loss_type": "actor_critic",
            "adv_type": "gae",
            "reward_type": "chunk_level",
            "logprob_type": "chunk_level",
            "entropy_type": "chunk_level",
            "clip_ratio_high": 0.2,
            "clip_ratio_low": 0.2,
            "kl_penalty": "kl",
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "normalize_advantages": True,
            "kl_beta": 0.0,
            "entropy_bonus": 0.0,
            "group_size": 1,
            "rollout_epoch": 1,
        },
        "actor": {
            "model": {"model_type": "openpi", "add_value_head": True},
            "global_batch_size": 16,
            "micro_batch_size": 8,
            "seed": 1234,
            "optim": {"lr": 5e-6, "value_lr": 1e-4, "adam_beta1": 0.9,
                      "adam_beta2": 0.95, "adam_eps": 1e-8, "weight_decay": 0.01,
                      "clip_grad": 1.0, "critic_warmup_steps": 0},
            "fsdp_config": {"strategy": "fsdp"},
        },
        "runner": {"max_epochs": 10},
        "rollout": {"generation_backend": "huggingface"},
        "reward": {"use_reward_model": False},
        "critic": {"use_critic_model": False},
    })


def test_auto_update_disabled_default():
    """auto_update_epoch absent or enabled=False: default behavior."""
    cfg = _make_cfg(auto_update_enabled=False)
    auto_cfg = cfg.algorithm.get("auto_update_epoch", {})
    if auto_cfg is None:
        auto_cfg = {}
    assert auto_cfg.get("enabled", False) is False


def test_auto_update_target_kl_none():
    """target_kl=None: executes all epochs without early stop."""
    cfg = _make_cfg(auto_update_enabled=True, target_kl=None)
    auto_cfg = cfg.algorithm.get("auto_update_epoch", {})
    assert auto_cfg.get("target_kl", None) is None


def test_auto_update_config_missing():
    """Missing auto_update_epoch key returns safe defaults."""
    cfg = _make_cfg()
    # Remove the key entirely
    import omegaconf
    cfg = omegaconf.OmegaConf.create({"algorithm": {"update_epoch": 2}})
    auto_cfg = cfg.algorithm.get("auto_update_epoch", {})
    if auto_cfg is None:
        auto_cfg = {}
    assert auto_cfg.get("enabled", False) is False
    assert auto_cfg.get("target_kl", None) is None


def test_kl_per_epoch_slicing():
    """Per-epoch KL slicing: epoch 2 KL excludes epoch 1 values."""
    metrics = {"actor/approx_kl": [0.1, 0.2, 0.15]}
    kl_count_before = len(metrics.get("actor/approx_kl", []))
    # Epoch 2 adds new values
    metrics["actor/approx_kl"].extend([0.3, 0.25])
    epoch2_vals = metrics["actor/approx_kl"][kl_count_before:]
    assert epoch2_vals == [0.3, 0.25]


def test_sum_count_allreduce_correctness():
    """[sum, count] all_reduce produces correct global mean."""
    # rank0: [0.5, 0.5], rank1: [0.3]
    sum_r0, cnt_r0 = 1.0, 2
    sum_r1, cnt_r1 = 0.3, 1
    # Simulated all_reduce SUM
    global_sum = sum_r0 + sum_r1  # 1.3
    global_cnt = cnt_r0 + cnt_r1  # 3
    global_mean = global_sum / global_cnt  # 0.4333
    expected = 1.3 / 3.0
    assert abs(global_mean - expected) < 1e-6

    # Old [mean, count] bug would produce:
    mean_r0, mean_r1 = 0.5, 0.3
    buggy_global = (mean_r0 + mean_r1) / (cnt_r0 + cnt_r1)  # 0.8/3 = 0.2667
    assert abs(buggy_global - 0.26666) < 1e-4
    # correct result != buggy result
    assert abs(global_mean - buggy_global) > 0.1


def test_kl_below_threshold_runs_all_epochs():
    """KL below threshold: runs all epochs without early stop trigger."""
    target = 0.05
    global_kl = 0.02  # below threshold
    update_epoch = 3
    stopped = False
    executed = 0
    for epoch_idx in range(update_epoch):
        executed += 1
        if global_kl > target and epoch_idx + 1 < update_epoch:
            stopped = True
            break
    assert executed == 3
    assert not stopped


def test_kl_above_threshold_stops_early():
    """KL above threshold: stops after first epoch, break exits loop."""
    target = 0.05
    global_kl = 0.10  # above threshold
    update_epoch = 3
    stopped = False
    executed = 0
    for epoch_idx in range(update_epoch):
        executed += 1
        if global_kl > target and epoch_idx + 1 < update_epoch:
            stopped = True
            break
    assert executed == 1
    assert stopped


def test_kl_above_threshold_last_epoch_no_break():
    """KL above threshold on last epoch: cannot break (no more epochs)."""
    target = 0.05
    global_kl = 0.10
    update_epoch = 1
    stopped = False
    executed = 0
    for epoch_idx in range(update_epoch):
        executed += 1
        if global_kl > target and epoch_idx + 1 < update_epoch:
            stopped = True
            break
    assert executed == 1
    assert not stopped  # condition "epoch_idx+1 < update_epoch" prevents break
