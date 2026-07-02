# Agent.OnActionReceived 输入来源

`OnActionReceived(ActionBuffers actionBuffers)` 中的 `actionBuffers` **不是你在代码里手动传进去的**，而是 ML-Agents 框架在每一帧决策循环里自动生成，再通过 `ActuatorManager` 转发给你的 `OnActionReceived` 方法。

以 `Ball3DAgent` 为例，你读取的是：

```csharp
actionBuffers.ContinuousActions[0]  // z 轴旋转
actionBuffers.ContinuousActions[1]  // x 轴旋转
```

这两个值来自 **Behavior Parameters** 组件上配置的 **Action Spec**（2 个连续动作）。

---

## 整体流程

每个 Academy Step 大致按以下顺序执行：

```
DecisionRequester.RequestDecision()
        ↓
Agent.SendInfoToBrain()  →  CollectObservations() 收集观测
        ↓
Policy.RequestDecision()  →  把观测发给「决策者」
        ↓
Academy.DecideAction()  →  Policy.DecideAction()  →  得到 ActionBuffers
        ↓
ActuatorManager.UpdateActions(actions)  →  存入 StoredActions
        ↓
Agent.AgentStep()  →  ActuatorManager.ExecuteActions()
        ↓
VectorActuator.OnActionReceived()  →  Ball3DAgent.OnActionReceived(actionBuffers)
```

---

## actionBuffers 的具体来源（取决于运行模式）

由 **Behavior Parameters** 组件上的 **Behavior Type** 决定使用哪种 Policy：

| 模式 | Policy | actionBuffers 从哪来 |
|------|--------|----------------------|
| **Default + 连接 Python 训练** | `RemotePolicy` | Python 端 PPO 等算法根据观测算出动作，经 Communicator 发回 Unity |
| **Default / Inference Only + 挂了 .onnx 模型** | `BarracudaPolicy` | 神经网络（Barracuda）对观测做推理，输出动作 |
| **Heuristic Only**（或 Default 且无模型、未连训练） | `HeuristicPolicy` | 调用你写的 `Heuristic()`，把键盘输入写入 `actionsOut`，再作为动作传出 |

### 1. 训练时（Python）

`RemotePolicy.DecideAction()` 从 Python 取回动作：

```csharp
m_Communicator?.DecideBatch();
var actions = m_Communicator?.GetActions(behaviorName, agentId);
```

### 2. 推理时（已训练 .onnx 模型）

`BarracudaPolicy.DecideAction()` 用模型推理：

```csharp
m_ModelRunner?.DecideBatch();
m_LastActionBuffer = m_ModelRunner.GetAction(m_AgentId);
```

### 3. 手动控制（Heuristic）

`HeuristicPolicy.DecideAction()` 调用你的 `Heuristic()`：

```csharp
m_ActuatorManager.ApplyHeuristic(m_ActionBuffers);  // 内部会调用 Ball3DAgent.Heuristic()
```

在 `Ball3DAgent.Heuristic()` 中写入的值：

```csharp
continuousActionsOut[0] = -Input.GetAxis("Horizontal");
continuousActionsOut[1] = Input.GetAxis("Vertical");
```

会进入同一个 `ActionBuffers`，随后在 `OnActionReceived` 中被读取。

---

## 谁负责把动作传给你的方法？

`Agent` 初始化时会创建 `AgentVectorActuator`，把你的 `Ball3DAgent` 包装成 Actuator。执行阶段：

1. `Agent.DecideAction()` 从 Policy 拿到 `ActionBuffers`
2. `ActuatorManager.UpdateActions()` 存起来
3. `ActuatorManager.ExecuteActions()` 按 `ActionSpec` 切片，调用 `VectorActuator.OnActionReceived()`
4. 最终调用你的 `Ball3DAgent.OnActionReceived(actionBuffers)`

---

## 与 Heuristic() 的关系

| 方法 | 角色 |
|------|------|
| `Heuristic()` | 你**写出**动作（输入端） |
| `OnActionReceived()` | 你**读取并执行**动作（输出端） |

在 Heuristic 模式下，动作路径是：`Heuristic()` 填值 → Policy 返回 → `OnActionReceived()` 使用。

在训练/推理模式下，动作由外部算法或神经网络产生，你只负责在 `OnActionReceived()` 里消费。

---

## 触发时机

场景里通常挂有 **Decision Requester** 组件，每 `DecisionPeriod` 个 Academy Step 调用一次 `RequestDecision()`。

若 `TakeActionsBetweenDecisions` 为 true，中间步会调用 `RequestAction()`，重复执行**上一次**的动作，而不会重新决策。

---

## 总结

`actionBuffers` 是 ML-Agents 根据当前 **Behavior Type**（训练 / 模型推理 / 键盘 Heuristic）自动生成的动作数组，经 `ActuatorManager` 传入你的 `OnActionReceived`。你只需要读取并用来控制 Agent 行为（例如倾斜平板），无需自己构造或传入该参数。
