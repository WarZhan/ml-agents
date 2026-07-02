# Unity ML-Agents 训练配置（config/）规制说明

> `config/` 下的 YAML 不是自动生成的，而是**手写的训练配置**。所谓"生成规则"其实是它们必须遵守的
> **目录/命名约定**与 **schema（结构层级）**。本文基于仓库 `config/` 下全部配置文件梳理而成。

---

## 一、目录与命名规则

```
config/<算法>/<行为名>.yaml
```

- **第一层子目录 = 算法（trainer_type）**：`ppo/`、`sac/`、`poca/`、`imitation/`
  （模仿学习本质仍是 ppo/sac + BC/GAIL 奖励）。
- **文件名 = 对应示例 / 行为**：如 `3DBall.yaml`；变体加后缀说明用途，例如：
  - `3DBall_randomize.yaml`（域随机化）
  - `WallJump_curriculum.yaml`（课程学习）
  - `PyramidsRND.yaml`（RND 探索）
- **最关键的绑定规则**：文件里 `behaviors:` 下的**键名必须与 Unity 中 `BehaviorParameters` 组件的
  Behavior Name 完全一致**。例如 `WallJump_curriculum.yaml` 里是 `BigWallJump` / `SmallWallJump`
  两个键，正是因为 `WallJumpAgent` 运行时会切换这两个行为名。训练器靠这个名字把 Python 端的 trainer
  与 Unity 端的 Agent 对接起来。

---

## 二、顶层结构（schema 根节点）

一个配置文件最多有这几个顶层键，只有 `behaviors` 是必需的：

```yaml
behaviors:                 # 必需：每个 Behavior 一套训练参数
environment_parameters:    # 可选：环境参数（课程学习 / 域随机化）
default_settings:          # 可选：所有 behavior 的默认值（可被单独覆盖）
env_settings:              # 可选：环境可执行文件、并行环境数等
engine_settings:           # 可选：time_scale、分辨率等 Unity 引擎设置
checkpoint_settings:       # 可选：run-id、输出目录等（通常用命令行覆盖）
torch_settings:            # 可选：device (cpu/cuda)
```

> 命令行参数（如 `--run-id`）会覆盖文件中的对应项，所以示例文件里通常省略 `checkpoint_settings`。

---

## 三、单个 behavior 的字段规则

以 `config/ppo/3DBall.yaml` 为骨架：

```yaml
behaviors:
  3DBall:                       # ← 必须匹配 Unity Behavior Name
    trainer_type: ppo           # ppo | sac | poca
    hyperparameters: {...}      # 算法超参数（因 trainer_type 而异）
    network_settings: {...}     # 网络结构
    reward_signals: {...}       # 奖励信号（至少要有 extrinsic）
    keep_checkpoints: 5
    max_steps: 500000
    time_horizon: 1000
    summary_freq: 12000
```

### 1) `hyperparameters` —— 随算法不同而不同

- **公共**：`batch_size`、`buffer_size`、`learning_rate`、`learning_rate_schedule`(linear/constant)
- **PPO / POCA 专有**：`beta`、`epsilon`、`lambd`、`num_epoch`
- **SAC 专有**：`tau`、`buffer_init_steps`、`steps_per_update`、`init_entcoef`、
  `save_replay_buffer`、`reward_signal_steps_per_update`

> 对照 `config/ppo/3DBall.yaml` 与 `config/sac/3DBall.yaml` 即可看到差异。

### 2) `network_settings` —— 网络结构

```yaml
network_settings:
  normalize: true          # 是否归一化向量观测（连续控制类常开）
  hidden_units: 128
  num_layers: 2
  vis_encode_type: simple  # 视觉观测编码器
  memory:                  # 可选：启用 LSTM（记忆型任务，如 Hallway）
    sequence_length: 64
    memory_size: 128
```

### 3) `reward_signals` —— 奖励信号（可叠加多个）

```yaml
reward_signals:
  extrinsic:               # 环境奖励，几乎必有
    gamma: 0.99
    strength: 1.0
  curiosity:               # 好奇心（内在奖励，稀疏奖励任务用）
    strength: 0.02
  rnd:                     # Random Network Distillation（见 PyramidsRND）
    strength: 0.01
  gail:                    # 生成对抗模仿学习，需要 demo_path
    strength: 0.01
    demo_path: Project/Assets/ML-Agents/Examples/Pyramids/Demos/ExpertPyramid.demo
```

### 4) 可选功能块

- **`behavioral_cloning`（BC，模仿学习）**：需 `demo_path` + `strength` + `steps`
  （见 `config/imitation/Pyramids.yaml`）。
- **`self_play`（自对弈，对抗类）**：`save_steps`、`team_change`、`swap_steps`、`window`、
  `play_against_latest_model_ratio`、`initial_elo`（见 `config/poca/SoccerTwos.yaml`）。

```yaml
self_play:
  save_steps: 50000
  team_change: 200000
  swap_steps: 2000
  window: 10
  play_against_latest_model_ratio: 0.5
  initial_elo: 1200.0
```

---

## 四、`environment_parameters` —— 两种模式

这里的**键名必须匹配 C# 中 `EnvironmentParameters.GetWithDefault("键名", 默认值)` 的字符串**
（例如 `Ball3DAgent` 里的 `mass` / `scale`，`WallJumpAgent` 里的 `big_wall_height`）。

### 模式 A：Sampler（域随机化）—— 每回合随机采样

```yaml
environment_parameters:
  mass:
    sampler_type: uniform          # uniform | gaussian | multirangeuniform
    sampler_parameters:
      min_value: 0.5
      max_value: 10
```

### 模式 B：Curriculum（课程学习）—— 按训练进度分级递进

```yaml
environment_parameters:
  big_wall_height:
    curriculum:
      - name: Lesson0               # 列表项，'-' 必须有
        completion_criteria:
          measure: progress         # progress | reward
          behavior: BigWallJump     # 依据哪个 behavior 的表现
          threshold: 0.1            # 达标阈值 → 进入下一课
          min_lesson_length: 100
          signal_smoothing: true
        value:                      # 该课的参数值，可以是定值或 sampler
          sampler_type: uniform
          sampler_parameters:
            min_value: 0.0
            max_value: 4.0
      - name: Lesson3
        value: 8.0                  # 最后一课通常给定值、无 completion_criteria
```

---

## 五、按需求速查表

| 需求 | 放哪个块 | 参考文件 |
| --- | --- | --- |
| 基础 PPO/SAC 训练 | `hyperparameters` + `reward_signals.extrinsic` | `ppo/3DBall.yaml`、`sac/3DBall.yaml` |
| 记忆任务（LSTM） | `network_settings.memory` | `ppo/Hallway.yaml` |
| 稀疏奖励探索 | `reward_signals.curiosity` / `rnd` | `ppo/PyramidsRND.yaml` |
| 模仿学习 | `behavioral_cloning` + `reward_signals.gail` | `imitation/Pyramids.yaml` |
| 多智能体对抗 | `trainer_type: poca` + `self_play` | `poca/SoccerTwos.yaml` |
| 域随机化 | `environment_parameters` + `sampler_type` | `ppo/3DBall_randomize.yaml` |
| 课程学习 | `environment_parameters` + `curriculum` | `ppo/WallJump_curriculum.yaml` |

---

## 六、核心规律总结

1. **目录按算法分，文件名按示例分**；变体用后缀标明特性。
2. **两个"名字必须对齐"的硬规则**：
   - `behaviors` 的键 ↔ Unity `BehaviorParameters` 的 Behavior Name；
   - `environment_parameters` 的键 ↔ C# 里 `GetWithDefault(...)` 的字符串。
3. **结构是"公共骨架 + 按需追加功能块"**：`hyperparameters` / `network_settings` /
   `reward_signals` 是骨架，`memory` / `self_play` / `behavioral_cloning` / `curriculum`
   等按任务特性叠加。

---

## 七、训练命令示例

```bash
mlagents-learn config/ppo/3DBall.yaml --run-id=first3DBallRun
```

- `--run-id`：本次运行的唯一标识，结果输出到 `results/<run-id>/`。
- `--resume` / `--force`：续训 / 覆盖同名 run。
- `--initialize-from=<run-id>`：从已有 run 的权重初始化。
