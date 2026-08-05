# Migration Log

> 从 `rebase-main` (codex/behavior-rebase-main) 往 `clean` (upstream/main) 分批迁移 RL 功能。
> 来源仓库: `/mnt/public/juchuanqi/repo/RLinf-zcy/RLinf-rebase-main`
> 目标仓库: `/mnt/public/juchuanqi/repo/RLinf-clean`
> 开始日期: 2026-08-05

---

## PR 1: RL 核心基础设施

分支: `feat/rl-core-infra` (待创建)

### 改动记录

| # | 日期 | 文件 | 改动 | 说明 | commit |
|---|---|---|---|---|---|
| 1 | 2026-08-05 | `rlinf/algorithms/utils.py` | +56 | `_expand_singleton_action_dim` + `action_level` reshape + GAE expand block | `fab0b616` |
| 2 | 2026-08-05 | `rlinf/workers/actor/fsdp_actor_worker.py` | +34/-1 | `auto_update_epoch` KL-aware early stopping | `a9b5e735` |

### 决定不移植

| 文件 | 原因 |
|---|---|
| `losses.py` (compute_explained_variance) | 上游已用 `compute_critic_explained_variance_stats` 重构，方案更优 |

---

## PR 2: BEHAVIOR 环境 + 数据管道

分支: `feat/behavior-env` (待创建)

### 改动记录

| # | 日期 | 文件 | 改动 | 说明 | commit |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

---

## PR 3: OpenPI Flow-SDE 噪声

分支: `feat/openpi-flow-sde` (待创建)

### 改动记录

| # | 日期 | 文件 | 改动 | 说明 | commit |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

---

## PR 4: BEHAVIOR 评估 + 指标

分支: `feat/behavior-eval` (待创建)

### 改动记录

| # | 日期 | 文件 | 改动 | 说明 | commit |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

---

## PR 5: 工具链 + 配置 + 测试 + 文档

分支: `feat/behavior-tooling` (待创建)

### 改动记录

| # | 日期 | 文件 | 改动 | 说明 | commit |
|---|---|---|---|---|---|
| — | — | — | — | — | — |
