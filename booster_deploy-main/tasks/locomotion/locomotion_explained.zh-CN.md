# `tasks/locomotion/locomotion.py` 逐步解释

本文解释文件：

- [tasks/locomotion/locomotion.py](/home/zbc/booster_t1_real/booster_deploy-main/tasks/locomotion/locomotion.py)
- 以及它的任务注册文件 [tasks/locomotion/__init__.py](/home/zbc/booster_t1_real/booster_deploy-main/tasks/locomotion/__init__.py)

这份 `locomotion.py` 本身**不是完整控制器**，它主要负责两件事：

1. 定义 walking policy 的**观测构造和动作推理**
2. 定义 K1 / T1 walking 任务的**配置**

真正的运行入口在：

- `python scripts/deploy.py --task t1_walk`

真正执行控制循环的是：

- `booster_deploy/controllers/booster_robot_controller.py`（真机 / Webots）
- `booster_deploy/controllers/mujoco_controller.py`（MuJoCo 仿真）

---

## 1. 文件整体结构

`locomotion.py` 可以分成 4 块：

1. 导入依赖
2. `LocomotionPolicy`：定义 walking 策略
3. `LocomotionPolicyCfg`：定义策略配置
4. `K1WalkControllerCfg` / `T1WalkControllerCfg`：定义机器人和任务配置

---

## 2. 导入部分

代码开头：

```python
from __future__ import annotations
from dataclasses import MISSING
import os
import torch
```

作用：

- `__future__.annotations`：让类型注解延迟求值，减少前向引用问题
- `MISSING`：给配置类里的“必须填写字段”做占位
- `os`：处理模型路径
- `torch`：加载 TorchScript 模型、构造张量、推理

然后导入框架里的基础类：

```python
from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, PolicyCfg, VelocityCommandCfg
)
```

作用：

- `Policy`：所有策略类的基类
- `BaseController`：策略运行时拿机器人状态、命令输入、时间步等都要靠它
- `ControllerCfg / PolicyCfg / VelocityCommandCfg`：配置系统的基类

再往下：

```python
from booster_deploy.robots.booster import K1_CFG, T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math
```

作用：

- `K1_CFG` / `T1_23DOF_CFG`：机器人原始配置模板
- `configclass`：把普通 Python 类包装成可递归配置对象
- `lab_math`：提供 `quat_apply_inverse` 等姿态/向量变换函数

---

## 3. `LocomotionPolicy`：walking 策略本体

类定义：

```python
class LocomotionPolicy(Policy):
    """walking policy with observation history."""
```

它的核心职责是：

1. 加载训练好的 walking 模型
2. 从当前机器人状态构造 observation
3. 维护 observation history
4. 输出每一步的关节目标 `dof_targets`

---

## 4. `__init__`：初始化策略

### 4.1 保存配置和机器人句柄

```python
self.cfg = cfg
self.robot = controller.robot
```

作用：

- `cfg`：拿到策略配置，比如模型路径、`action_scale`、关节名列表
- `self.robot`：拿到机器人对象，后续从里面读取关节角、关节速度、IMU 相关状态

### 4.2 解析模型路径

```python
policy_path = self.cfg.checkpoint_path
if not os.path.isabs(policy_path):
    policy_path = os.path.join(self.task_path, self.cfg.checkpoint_path)
```

作用：

- 如果用户给的是绝对路径，直接用
- 如果给的是相对路径，就按“当前任务目录”拼成完整路径

这能让任务目录自带模型，例如：

```python
models/t1_walk.pt
```

### 4.3 加载 TorchScript 模型

```python
self._model: torch.jit.ScriptModule = torch.jit.load(
    policy_path, map_location="cpu")
self._model.eval()
```

作用：

- 加载导出的 TorchScript walking 策略
- `map_location="cpu"`：部署默认放在 CPU 上
- `eval()`：进入推理模式，关闭训练态行为

### 4.4 读配置参数

```python
self.actor_obs_history_length = cfg.actor_obs_history_length
self.action_scale = cfg.action_scale
```

作用：

- `actor_obs_history_length`：模型不是只看当前一帧，而是看一段历史
- `action_scale`：控制动作输出放大/缩小多少

### 4.5 初始化缓存

```python
self.obs_history = None
self.last_action = torch.zeros(
    len(self.cfg.policy_joint_names), dtype=torch.float32)
```

作用：

- `obs_history`：后面第一次推理时才按真实维度初始化
- `last_action`：上一拍动作，会被拼进 observation

上一拍动作进 observation 的好处是：

- 给策略一点“自身控制输出的记忆”
- 减少动作突变

### 4.6 建立 real joint -> policy joint 映射

```python
self.real2sim_joint_map = torch.tensor([
    self.robot.cfg.joint_names.index(name)
    for name in self.cfg.policy_joint_names
], dtype=torch.long)
```

这是这份代码里非常关键的一步。

作用：

- `self.robot.cfg.joint_names` 是机器人完整关节顺序
- `policy_joint_names` 是模型真正控制/观测的关节顺序
- 这段代码把“策略关节顺序”映射成“真实机器人关节索引”

后面：

- 取 observation 里的关节位置/速度
- 把 action 写回到完整 `dof_targets`

都要靠这个映射。

---

## 5. `reset()`

```python
def reset(self) -> None:
    """Initialize policy state."""
    pass
```

当前没有额外重置逻辑。

这意味着：

- policy 本身是无显式内部状态的
- 真正的历史缓存 `obs_history` 在第一次 `inference()` 里懒初始化

如果后面想在每次启动/重启时清空 history，这里可以补。

---

## 6. `compute_observation()`：构造当前观测

这是策略最核心的部分之一。

### 6.1 读取机器人状态

```python
dof_pos = self.robot.data.joint_pos
dof_vel = self.robot.data.joint_vel
base_quat = self.robot.data.root_quat_w
base_ang_vel = self.robot.data.root_ang_vel_b
```

作用：

- `joint_pos`：所有关节角
- `joint_vel`：所有关节角速度
- `root_quat_w`：机器人基座姿态（世界系四元数）
- `root_ang_vel_b`：机器人基座角速度（机体系）

### 6.2 把重力投影到机体系

```python
gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)
```

这一步很重要。

含义：

- 世界系里重力方向恒定向下 `[0, 0, -1]`
- 但机器人身体会前倾、后仰、侧倾
- 所以要把重力向量转换到机器人机体系

策略通过这个量就能感知：

- 我现在是不是前倾了
- 是不是侧倒了
- 身体姿态是否稳定

### 6.3 安全保护：跌倒检测

```python
if self.cfg.enable_safety_fallback:
    if projected_gravity[2] > -0.5:
        ...
        self.controller.stop()
```

作用：

- `projected_gravity[2]` 接近 `-1` 时，说明身体“竖直”
- 如果它大于 `-0.5`，通常表示机器人已经明显倾倒

此时直接停控制器，避免继续输出动作。

这不是严格物理证明，而是一个很实用的跌倒近似判断。

### 6.4 读取速度命令

```python
lin_vel_x = self.controller.vel_command.lin_vel_x
lin_vel_y = self.controller.vel_command.lin_vel_y
ang_vel_yaw = self.controller.vel_command.ang_vel_yaw
```

作用：

- 从遥控器 / 键盘 / 仿真输入里读当前命令
- 这 3 个量分别是：
  - 前后速度
  - 左右速度
  - 转向角速度

### 6.5 根据映射抽取策略关节

```python
default_joint_pos_sim = self.robot.default_joint_pos
mapped_default_pos = default_joint_pos_sim[self.real2sim_joint_map]
mapped_dof_pos = dof_pos[self.real2sim_joint_map]
mapped_dof_vel = dof_vel[self.real2sim_joint_map]
```

作用：

- `robot.default_joint_pos` 是完整机器人所有关节默认位
- 策略不一定控制所有关节
- 所以先根据 `real2sim_joint_map` 抽出“策略真正关心的那些关节”

### 6.6 拼接 observation

```python
obs = torch.cat([
    base_ang_vel,
    projected_gravity,
    torch.tensor([lin_vel_x, lin_vel_y, ang_vel_yaw], dtype=torch.float32),
    (mapped_dof_pos - mapped_default_pos) * 1.0,
    mapped_dof_vel * self.cfg.obs_dof_vel_scale,
    self.last_action * 1.0
], dim=0)
```

这份 observation 结构是：

1. `base_ang_vel`：3 维  
   机器人身体角速度

2. `projected_gravity`：3 维  
   姿态信息

3. `commands`：3 维  
   期望速度命令

4. `joint_pos - default_pos`：`num_action` 维  
   关节偏离默认姿态多少

5. `joint_vel`：`num_action` 维  
   关节速度

6. `last_action`：`num_action` 维  
   上一拍动作

对 T1 来说，`num_action = 21`，所以总维度是：

```text
3 + 3 + 3 + 21 + 21 + 21 = 72
```

这个文件本身没显式写“72”，但结构就是这样。

### 为什么用 `joint_pos - default_pos`

因为策略通常更容易学：

```text
我偏离参考姿态多少
```

而不是直接学绝对关节角。

---

## 7. `inference()`：从 observation 到关节目标

### 7.1 先构造当前 observation

```python
obs = self.compute_observation()
```

### 7.2 第一次调用时初始化历史缓存

```python
if self.obs_history is None:
    self.obs_history = torch.zeros(
        self.actor_obs_history_length,
        obs.numel(),
        dtype=torch.float32
    )
```

如果 `actor_obs_history_length = 10`，那这里就会建立一个：

```text
10 x obs_dim
```

的历史缓冲。

### 7.3 维护 observation history

```python
self.obs_history = self.obs_history.roll(shifts=-1, dims=0)
self.obs_history[-1] = obs.clamp(-100.0, 100.0)
```

作用：

- `roll(-1)`：把历史往前推一格
- 最后一格写入最新 observation

这样历史队列始终保存最近 N 帧。

`clamp(-100, 100)` 是个简单保护，避免输入数值爆掉。

### 7.4 调模型推理

```python
with torch.no_grad():
    action = self._model(self.obs_history.flatten()).squeeze(0)
    action = torch.clamp(action, -100.0, 100.0)
```

作用：

- 把 `history_len x obs_dim` 展平，喂给 TorchScript 模型
- 模型输出动作
- 再做一次裁剪

这里说明模型结构不是 RNN，而更像：

```text
把多帧 observation 展平后送入 MLP
```

### 7.5 保存上一拍动作

```python
self.last_action = action.clone()
```

下一拍 observation 会把它拼进去。

### 7.6 构造完整关节目标

```python
default_joint_pos = self.robot.default_joint_pos
dof_targets = default_joint_pos.clone()
```

先从机器人默认姿态开始。

然后：

```python
dof_targets.scatter_reduce_(
    0,
    self.real2sim_joint_map,
    action * self.action_scale,
    reduce='sum')
```

作用：

- 只在 `policy_joint_names` 对应的那些关节上加动作
- 其他关节保持默认位

也就是说，这个 walking policy 不是“给出 23 个关节的绝对目标角”，而是：

```text
默认姿态 + 某些关节的相对动作偏移
```

### 为什么用 `scatter_reduce_`

因为 `policy_joint_names` 的顺序不一定等于完整机器人关节顺序。  
这一步是按照索引把动作放回完整的关节向量。

---

## 8. `LocomotionPolicyCfg`

```python
@configclass
class LocomotionPolicyCfg(PolicyCfg):
    constructor = LocomotionPolicy
    checkpoint_path: str = MISSING
    actor_obs_history_length: int = 10
    action_scale: float = 0.25
    obs_dof_vel_scale: float = 1.0
    policy_joint_names: list[str] = MISSING
```

这是 walking policy 的配置类。

字段含义：

- `constructor`：告诉框架这个配置该实例化成哪个 Policy 类
- `checkpoint_path`：模型文件路径
- `actor_obs_history_length`：历史帧长度
- `action_scale`：动作缩放
- `obs_dof_vel_scale`：关节速度的 observation 缩放
- `policy_joint_names`：策略关心的关节名称顺序

---

## 9. `K1WalkControllerCfg`

这一段定义 K1 机器人的 walking 任务配置。

重点是：

```python
robot = K1_CFG.replace(...)
```

它不是凭空新建一台机器人，而是在 `K1_CFG` 模板上：

- 改默认姿态
- 改 walking 用的 PD 参数

### 这里改了什么

1. `default_joint_pos`  
   walking 初始参考姿态

2. `joint_stiffness`  
   walking 运行期刚度

3. `joint_damping`  
   walking 运行期阻尼

### `vel_command`

```python
vel_command: VelocityCommandCfg = VelocityCommandCfg(
    vx_max=1.0,
    vy_max=1.0,
    vyaw_max=1.0,
)
```

作用：

- 限制命令最大值

### `policy`

```python
policy: LocomotionPolicyCfg = LocomotionPolicyCfg(...)
```

这里把 K1 对应的：

- `obs_dof_vel_scale`
- `policy_joint_names`

写进去。

---

## 10. `T1WalkControllerCfg`

和 K1 那段同理，只是换成 T1。

### 10.1 `robot = T1_23DOF_CFG.replace(...)`

这一步定义 T1 的 walking 用：

- 默认姿态
- PD 刚度
- PD 阻尼

也就是说，**`locomotion.py` 不只是 policy 文件，它同时也是 walking 任务配置文件。**

### 10.2 `vel_command`

同样定义最大速度命令范围：

```python
vx_max=1.0
vy_max=1.0
vyaw_max=1.0
```

### 10.3 `policy`

这里给 T1 指定：

- `obs_dof_vel_scale = 1.0`
- `policy_joint_names = [...]`

T1 的 `policy_joint_names` 有 21 个关节，包含：

- 肩
- 腰
- 手臂
- 髋
- 膝
- 踝

这也是为什么 `locomotion` 相比某些只控腿的策略，会更稳一些：  
它把上半身姿态也一并纳入控制。

---

## 11. `__init__.py` 里做了什么

文件：

[tasks/locomotion/__init__.py](/home/zbc/booster_t1_real/booster_deploy-main/tasks/locomotion/__init__.py)

### 11.1 定义最终部署用配置

```python
class T1WalkControllerCfg1(T1WalkControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/t1_walk.pt"
```

作用：

- 继承 `T1WalkControllerCfg`
- 只补一件事：把模型路径固定成任务目录里的 `models/t1_walk.pt`

### 11.2 注册任务

```python
register_task("t1_walk", T1WalkControllerCfg1())
```

这一步非常关键。

它的作用是：

- 当你运行：

  ```bash
  python scripts/deploy.py --task t1_walk
  ```

- `scripts/deploy.py` 会在 registry 里按名字查
- 找到这里注册进去的 `T1WalkControllerCfg1`

也就是说，**`t1_walk` 这个任务名就是在这里诞生的。**

---

## 12. `locomotion.py` 在整个系统里的运行位置

如果把整套部署流程串起来，`locomotion.py` 在里面大概处于这个位置：

1. `scripts/deploy.py` 解析命令行
2. 导入 `tasks.*`
3. `tasks/locomotion/__init__.py` 注册 `t1_walk`
4. 根据 `--task t1_walk` 取到 `T1WalkControllerCfg1`
5. 创建控制器：
   - 真机：`BoosterRobotPortal`
   - MuJoCo：`MujocoController`
6. 控制器在每个周期里：
   - `update_state()`
   - `policy_step()`
   - `ctrl_step()`
7. `policy_step()` 内部就会调用这里的：
   - `LocomotionPolicy.compute_observation()`
   - `LocomotionPolicy.inference()`

所以你可以把 `locomotion.py` 理解成：

```text
walking 任务的“策略定义 + 配置定义”
```

而不是完整运行入口。

---

## 13. 为什么 `locomotion` 往往比较稳

从这份代码本身看，它稳定的原因主要有这些：

1. **用 observation history**
   - 不是只看一帧，而是看最近 10 帧

2. **把 projected gravity 放进 observation**
   - 姿态信息比较直接

3. **把 last_action 放进 observation**
   - 动作更连续

4. **用默认姿态 + 相对动作**
   - 比直接输出绝对关节角更稳

5. **`action_scale = 0.25`**
   - 动作幅度比较保守

6. **控制上不只管腿，还管上半身若干关节**
   - 更容易维持整体姿态

当然，真正的“稳”还依赖：

- `booster_robot_controller.py` 的 startup 插值
- `prepare_state`
- 运行期 PD
- 手柄/键盘输入

这些不在 `locomotion.py` 里，但和它一起构成了完整系统。

---

## 14. 读这份文件时最值得盯住的点

如果你之后想调这份 walking 任务，我建议优先关注这些位置：

### 想改 observation 结构
看：

- `compute_observation()`

### 想改历史长度
看：

- `LocomotionPolicyCfg.actor_obs_history_length`

### 想改动作保守程度
看：

- `LocomotionPolicyCfg.action_scale`

### 想改 walking 默认姿态 / PD
看：

- `T1WalkControllerCfg.robot = T1_23DOF_CFG.replace(...)`

### 想换 walking 模型
看：

- `tasks/locomotion/__init__.py`

里的：

```python
self.policy.checkpoint_path = "models/t1_walk.pt"
```

---

## 15. 一句话总结

`locomotion.py` 做的事情可以概括成一句话：

**它定义了 walking 策略如何看机器人、如何出动作，以及 K1/T1 walking 任务该用哪套默认姿态、PD 和模型。**

如果你愿意，下一步我还可以继续给你补一份：

- **`booster_robot_controller.py` 与 `locomotion.py` 如何配合运行** 的说明文档  
这样你就能把“策略层”和“控制器层”完整串起来。  
