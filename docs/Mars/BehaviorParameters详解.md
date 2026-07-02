# BehaviorParameters 详细参数理解

> `BehaviorParameters` 是每个 `Agent` GameObject 上**必挂**的核心组件，负责描述"这个智能体的大脑长什么样"，
> 并在运行时据此**生成决策策略（Policy）**。它是 Unity 端与 Python 训练器 / ONNX 推理模型对接的关键枢纽。
>
> 源码：`com.unity.ml-agents/Runtime/Policies/BehaviorParameters.cs`、`BrainParameters.cs`、`BarracudaPolicy.cs`。
> 组件菜单：`ML Agents / Behavior Parameters`。

---

## 一、组件职责一句话

`BehaviorParameters` 本身**不做决策**，它只保存配置，并在 Agent 初始化时通过 `GeneratePolicy(...)`
按当前设置**选择并创建**三种策略之一：远程训练（Python）、本地推理（Barracuda 模型）、或启发式（`Heuristic()`）。

---

## 二、Inspector 面板字段总览

Inspector 中从上到下依次显示以下字段（顺序见 `BehaviorParametersEditor.OnInspectorGUI`）：

| Inspector 字段 | 序列化字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- | --- |
| **Behavior Name** | `m_BehaviorName` | string | `"My Behavior"` | 行为名，训练/推理的身份标识 |
| **Behavior Parameters** | `m_BrainParameters` | `BrainParameters` | — | 观测与动作的规格（见第四节） |
| **Model** | `m_Model` | `NNModel` | null | 推理用的神经网络模型（.onnx/.nn） |
| **Inference Device** | `m_InferenceDevice` | `InferenceDevice` | `Default` | 推理运行设备（CPU/GPU/Burst） |
| **Behavior Type** | `m_BehaviorType` | `BehaviorType` | `Default` | 决策来源模式（见第三节） |
| **Team Id** | `TeamId` | int | 0 | 队伍编号，自对弈 / 多队伍用 |
| **Use Child Sensors** | `m_UseChildSensors` | bool | true | 是否收集子物体上的传感器 |
| **Use Child Actuators** | `m_UseChildActuators` | bool | true | 是否收集子物体上的执行器 |
| **Observable Attribute Handling** | `m_ObservableAttributeHandling` | enum | `Ignore` | 如何扫描 `[Observable]` 特性 |

---

## 三、Behavior Type（决策来源模式）

`enum BehaviorType { Default, HeuristicOnly, InferenceOnly }`

- **Default（默认）**：自动选择，优先级为——
  1. 若连接了 Python 训练器（Communicator 开启）→ 走**远程训练**（`RemotePolicy`）；
  2. 否则若指定了 `Model` → 走**本地推理**（`BarracudaPolicy`）；
  3. 否则 → 走**启发式**（`HeuristicPolicy`，即调用 `Agent.Heuristic()`）。
- **HeuristicOnly**：无论如何都用 `Heuristic()`（键盘/手动控制），常用于调试与录制模仿学习演示。
- **InferenceOnly**：强制用 `Model` 本地推理；若未指定模型会抛异常。

决策流程对应源码：

```215:249:com.unity.ml-agents/Runtime/Policies/BehaviorParameters.cs
        internal IPolicy GeneratePolicy(ActionSpec actionSpec, ActuatorManager actuatorManager)
        {
            switch (m_BehaviorType)
            {
                case BehaviorType.HeuristicOnly:
                    return new HeuristicPolicy(actuatorManager, actionSpec);
                case BehaviorType.InferenceOnly:
                    {
                        if (m_Model == null)
                        {
                            var behaviorType = BehaviorType.InferenceOnly.ToString();
                            throw new UnityAgentsException(
                                $"Can't use Behavior Type {behaviorType} without a model. " +
                                "Either assign a model, or change to a different Behavior Type."
                            );
                        }
                        return new BarracudaPolicy(actionSpec, actuatorManager, m_Model, m_InferenceDevice, m_BehaviorName);
                    }
                case BehaviorType.Default:
                    if (Academy.Instance.IsCommunicatorOn)
                    {
                        return new RemotePolicy(actionSpec, actuatorManager, FullyQualifiedBehaviorName);
                    }
                    if (m_Model != null)
                    {
                        return new BarracudaPolicy(actionSpec, actuatorManager, m_Model, m_InferenceDevice, m_BehaviorName);
                    }
                    else
                    {
                        return new HeuristicPolicy(actuatorManager, actionSpec);
                    }
                default:
                    return new HeuristicPolicy(actuatorManager, actionSpec);
            }
        }
```

> 提示：训练时**不需要**手动切到 InferenceOnly；保持 Default 并运行 `mlagents-learn` 即可自动走远程训练。

---

## 四、Behavior Parameters（BrainParameters：观测与动作规格）

源码：`BrainParameters.cs`。它定义了策略的**输入（观测）**和**输出（动作）**形状。

### 4.1 Vector Observation（向量观测）

| 字段 | 序列化名 | 说明 |
| --- | --- | --- |
| **Space Size** | `VectorObservationSize` | `Agent.CollectObservations(VectorSensor)` 里写入的观测值个数。必须与实际写入数量一致，否则报错。 |
| **Stacked Vectors** | `NumStackedVectorObservations` | 把最近 N 帧观测拼接后送入网络（范围 1~50）。用于让无记忆网络感知"运动趋势"。 |

> 注意：如果观测来自 `RayPerceptionSensorComponent`、`CameraSensor`、`BufferSensor` 等**传感器组件**，
> 那部分观测**不计入** Space Size（由各自组件独立定义），Space Size 只统计 `CollectObservations` 手写的部分。

#### 案例：为什么 3DBall 的 Space Size 是 8

`Space Size` 必须等于 `CollectObservations` 中所有 `AddObservation` 展开后的**标量个数**。
其中 `AddObservation(float)` 贡献 **1** 维，`AddObservation(Vector3)` 贡献 **3** 维。看 `Ball3DAgent`：

```csharp
public override void CollectObservations(VectorSensor sensor)
{
    if (useVecObs)
    {
        sensor.AddObservation(gameObject.transform.rotation.z);              // 1
        sensor.AddObservation(gameObject.transform.rotation.x);              // 1
        sensor.AddObservation(ball.transform.position - gameObject.transform.position); // Vector3 → 3
        sensor.AddObservation(m_BallRb.velocity);                           // Vector3 → 3
    }
}
```

| 代码 | 含义 | 维度 |
| --- | --- | --- |
| `rotation.z` | 平台绕 Z 轴旋转分量 | 1 |
| `rotation.x` | 平台绕 X 轴旋转分量 | 1 |
| `ball.position - gameObject.position` | 球相对平台的位置（`Vector3`） | 3 |
| `m_BallRb.velocity` | 球的速度（`Vector3`） | 3 |
| | **合计** | **8** |

即 1 + 1 + 3 + 3 = **8**。相关变体对照：

- **Visual3DBall**：`useVecObs` 为 false，`CollectObservations` 不写任何值，改用相机传感器，故向量 Space Size = **0**。
- **3DBallHard**：去掉球的速度观测（少一个 `Vector3`），Space Size = **5**，用于演示"信息更少、任务更难"。

### 4.2 Actions（动作规格，`ActionSpec`）

新版把动作统一成 `ActionSpec`，**连续动作与离散动作可同时存在**：

| 字段 | 说明 |
| --- | --- |
| **Continuous Actions** | 连续动作的数量（float 向量长度）。例：3DBall = 2（X/Z 轴旋转）。 |
| **Discrete Branches** | 离散动作**分支数**；每个分支再填各自可选动作个数（Branch Size）。例：PushBlock = 1 个分支、7 个可选值。 |

> `VectorActionSize` / `VectorActionSpaceType` / `SpaceType` 均已 **`[Obsolete]` 弃用**，
> 仅为向后兼容保留，新工程一律用 `ActionSpec`（Continuous Actions + Discrete Branches）。

---

## 五、Model 与 Inference Device（推理相关）

- **Model（`NNModel`）**：训练导出的 `.onnx`/`.nn` 模型资产。仅在本地推理（Barracuda）时使用。
  运行时**不要直接赋值**该字段，应调用 `Agent.SetModel(behaviorName, model, device)`（`WallJump` 按墙高切换模型就是用它）。
- **Inference Device（`InferenceDevice`）**：推理设备，取值：

| 值 | 含义 |
| --- | --- |
| `Default` (0) | 默认，目前等同 Burst，未来可能变化 |
| `GPU` (1) | GPU 推理（Barracuda ComputePrecompiled） |
| `Burst` (2) | CPU + Burst 编译（**推荐的 CPU 方式**） |
| `CPU` (3) | 旧版纯 CPU，仅为兼容保留 |

> 一般小网络用 CPU/Burst 反而更快（省去 GPU 数据往返）；视觉大网络可尝试 GPU。

---

## 六、Team Id（队伍编号）

- 用于**多队伍 / 自对弈（Self-Play）**场景，区分同一 Behavior 下的不同阵营。
- 它会拼进"完全限定行为名"：

```205:208:com.unity.ml-agents/Runtime/Policies/BehaviorParameters.cs
        public string FullyQualifiedBehaviorName
        {
            get { return m_BehaviorName + "?team=" + TeamId; }
        }
```

- 示例：`AgentSoccer` 读取 `BehaviorParameters.TeamId` 判断自己属于 Blue(0) 还是 Purple(1)，
  并据此镜像出生点与朝向；训练时 `config/poca/SoccerTwos.yaml` 的 `self_play` 块依赖不同 TeamId 实现对抗自对弈。

---

## 七、Use Child Sensors / Use Child Actuators

- **Use Child Sensors**（`m_UseChildSensors`，默认 true）：为 true 时会收集**子 GameObject 上所有 SensorComponent**
  （如挂在子物体上的射线传感器、相机传感器），一并作为该 Agent 的观测来源。
- **Use Child Actuators**（`m_UseChildActuators`，默认 true）：类似地收集子物体上的 `ActuatorComponent`。
- **重要限制**：这两个开关**在 Agent 初始化之后再改无效**（源码注释明确说明 "changing this after the Agent has been initialized will not have any effect"）。

---

## 八、Observable Attribute Handling（`[Observable]` 特性扫描）

`enum ObservableAttributeOptions { Ignore, ExcludeInherited, ExamineAll }`

控制是否/如何用反射扫描 Agent 上标了 `[Observable]` 特性的字段/属性并自动转成观测：

| 值 | 行为 | 性能 |
| --- | --- | --- |
| **Ignore**（默认） | 完全忽略 `[Observable]`，若不用该特性则初始化最快 | 最快 |
| **ExcludeInherited** | 只扫描**当前类**声明的成员，忽略继承来的 | 折中 |
| **ExamineAll** | 扫描所有成员（含继承） | 最慢，启动更久 |

> 只有当你确实用 `[Observable]` 特性来声明观测时，才需要改成后两者。

---

## 九、运行时 API（脚本操作）

- 属性 `BehaviorName` / `Model` / `InferenceDevice` / `BehaviorType` 的 setter 都会触发 `UpdateAgentPolicy()`
  → `Agent.ReloadPolicy()`，即改动会即时重建策略。
- `bool IsInHeuristicMode()`：判断当前是否处于启发式模式（无模型且无通信）。
- 事件 `OnPolicyUpdated(bool isInHeuristicMode)`：策略更新时触发（internal）。
- **推荐**：运行时切换模型请用 `Agent.SetModel(string behaviorName, NNModel model, InferenceDevice device)`，
  不要直接赋值 `Model` 字段。

---

## 十、与训练配置（config YAML）的对接关系

- **Behavior Name ↔ config `behaviors` 的键**：二者**必须完全一致**，否则 `mlagents-learn` 找不到对应 trainer。
  例：Inspector 里 Behavior Name = `3DBall` ↔ `config/ppo/3DBall.yaml` 里 `behaviors: 3DBall:`。
- **Space Size / ActionSpec ↔ 网络输入输出**：加载 ONNX 模型时，编辑器会用 `BarracudaModelParamLoader.CheckModel`
  校验观测/动作维度是否与模型匹配，不匹配会在 Inspector 显示 Info/Warning/Error 提示。
- **Team Id ↔ `self_play`**：自对弈训练依赖不同 Team Id 组织对抗。

---

## 十一、常见坑与建议

1. **Behavior Name 拼写**：与 YAML 键、`SetModel` 的 behaviorName 三处必须一致（含大小写）。
2. **Space Size 对不上**：`CollectObservations` 实际写入数量必须等于 Space Size；用传感器组件的观测**不要**重复计入。
3. **InferenceOnly 未给模型**：会直接抛 `UnityAgentsException`。
4. **子传感器/执行器开关改动无效**：需在初始化前设置好 `UseChildSensors`/`UseChildActuators`。
5. **训练时保持 Default**：让它自动连接 Python；训练完把导出的模型拖到 Model 字段即可切换为本地推理。
6. **多模型切换**（如 WallJump）：用 `Agent.SetModel` 在运行时按情境换脑，而非改 Inspector。
