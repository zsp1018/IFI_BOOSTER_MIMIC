# BaseWalk 与 Locomotion 的差异说明

本文聚焦下面两个任务在 `booster_deploy-main` 体系里的核心差异，并说明哪些差异更容易导致：

- `base_walk` 更容易抖动
- `locomotion` 相对更稳

对比对象：

- `tasks/base_walk/base_walk.py`
- `tasks/locomotion/locomotion.py`

---

## 一句话总结

`locomotion` 更像一套原生的、整合好的全身 walking 策略：

- 21 个控制关节
- 72 维单帧观测
- 10 帧历史观测
- 与当前 deploy 控制栈同源

而 `base_walk` 更像迁移进来的 HTWK 风格策略：

- 12 个腿部关节
- 47 维单帧观测
- 没有 `obs_history`
- 带显式 `phase / gait_process`
- 对低层控制语义、噪声、时序误差更敏感

因此，`base_walk` 更容易出现抖动、发颤、接触不顺等现象。

---

## 1. 历史观测 vs 单帧观测

### Locomotion

`locomotion` 的模型不是只看当前一帧，而是看最近一段时间：

- 单帧 observation：72 维
- 历史长度：10
- 模型实际输入：`72 x 10 = 720`

对应代码：

- `tasks/locomotion/locomotion.py`
  - `self.actor_obs_history_length`
  - `self.obs_history`
  - `self.obs_history.flatten()`

### BaseWalk

`base_walk` 当前是单帧输入：

- 单帧 observation：47 维
- 没有 `obs_history`
- 直接 `obs.unsqueeze(0)` 推理

对应代码：

- `tasks/base_walk/base_walk.py`
  - 没有 `self.obs_history`
  - `self._model(obs.unsqueeze(0))`

### 为什么这会影响抖动

历史观测的优势是：

- 能看到最近状态变化趋势
- 能分辨瞬时噪声和真实失稳趋势
- 对 IMU / 关节速度噪声更鲁棒

单帧策略则更容易：

- 对某一帧异常值过度反应
- 造成关节来回修正
- 表现成抖动或发颤

这是 `locomotion` 更稳的最重要原因之一。

---

## 2. 21 关节全身控制 vs 12 关节腿部控制

### Locomotion

`locomotion` 控制 21 个关节，包含：

- 肩
- 腰
- 肘
- 髋
- 膝
- 踝

### BaseWalk

`base_walk` 只控制 12 个腿部关节：

- 左右髋 pitch / roll / yaw
- 左右膝
- 左右踝 pitch / roll

### 为什么这会影响抖动

全身控制意味着策略可以使用：

- 上半身摆动
- 腰部调整
- 手臂配合

来帮助平衡。

`base_walk` 只有腿可用，稳定手段更少，因此：

- 对地面接触误差更敏感
- 对重心偏差更敏感
- 更容易出现前倾、侧倾和抖动

---

## 3. 隐式节奏学习 vs 显式步态相位

### Locomotion

`locomotion` 没有显式 `phase` 输入，更依赖：

- 历史观测
- 动态状态趋势

去隐式学习 walking 节奏。

### BaseWalk

`base_walk` 明确维护：

- `gait_frequency`
- `gait_process`
- `phase = [cos(2pi p), sin(2pi p)]`

### 为什么这会影响抖动

显式 phase 的优点是：

- 节拍清晰
- 容易训练出节奏感很强的 gait

但它的代价是：

- 对接触时序偏差更敏感
- 对真实机器人和仿真的节奏误差更敏感
- 一旦“该抬脚时没抬起来”或“该落脚时没落稳”，策略会更容易不顺

这经常表现成：

- 步态僵
- 落脚突兀
- 局部关节抖动

---

## 4. 原始命令 vs 平滑命令

### Locomotion

`locomotion` 直接使用：

- `lin_vel_x`
- `lin_vel_y`
- `ang_vel_yaw`

### BaseWalk

`base_walk` 在推理前会先更新：

- `self.smoothed_commands`

通过限速变化的方式平滑命令。

### 这件事本身是不是坏事

不是。命令平滑通常是好事。

但它意味着 `base_walk` 的策略语义和 `locomotion` 已经不同了：

- `base_walk` 学到的是“平滑命令驱动下怎么走”
- `locomotion` 学到的是“直接速度命令驱动下怎么走”

如果训练和部署两边对这件事理解不完全一致，也可能带来动作差异。

---

## 5. 动作语义和动作幅度敏感性

### Locomotion

`locomotion` 当前使用：

- `action_scale = 0.25`

这会让动作偏移更保守。

### BaseWalk

`base_walk` 不同版本里，`action_scale` 曾经出现过更激进的设置。  
如果使用 `1.0`，会明显更容易导致：

- 目标角跳变大
- 电机追目标更猛
- 接触瞬间发力更硬

### 为什么这会影响抖动

同样一份策略输出，如果放大更多，真机或仿真就更容易：

- 抢修过度
- 触地产生冲击
- 表现成明显抖动

因此 `action_scale` 是一个非常敏感的“放大器”。

---

## 6. 低层控制语义是否同源

这是另一个特别重要的来源。

### Locomotion

`locomotion` 是围绕当前 `booster_deploy-main` 控制栈设计的：

- observation
- joint order
- default pose
- startup 链
- running PD
- 低层控制出口

都属于同一套体系。

### BaseWalk

`base_walk` 则更像是把 HTWK 那边的策略风格迁入当前框架：

- 训练逻辑来自另一套
- phase / gait_process 风格也来自另一套
- 原始 deploy 对低层控制还有自己的一些假设

这意味着它更容易出现“表面兼容、底层语义不完全一致”的情况。

### 为什么这会影响抖动

即使下面这些都对齐了：

- default pose
- stiffness
- damping
- action_scale

仍然可能因为这些更深层的差异而抖动：

- 接触模型不同
- actuator / torque clipping 语义不同
- 并联关节控制处理不同
- delay / friction / filtering 方式不同

---

## 7. 哪些差异最可能导致 BaseWalk 抖动

按优先级排序，我会这样看：

1. **没有 `obs_history`，单帧策略对噪声更敏感**
2. **只控制腿部 12 个关节，缺少上半身稳定辅助**
3. **显式 `phase / gait_process` 对时序误差更敏感**
4. **训练和 deploy 的底层控制语义不是完全同源**
5. **动作缩放 / 目标角语义更容易偏激**

---

## 8. 哪些差异本身不是 bug

下面这些差异并不天然等于“错”，但它们会改变行为：

- 用 `phase` 而不是 `obs_history`
- 用 `smoothed_commands` 而不是原始命令
- 只控制腿部而不控制上半身
- 使用不同的默认姿态基准

这些更像是“设计选择”，只是它们的鲁棒性边界不同。

---

## 9. 对调试的实际启发

如果目标是减少 `base_walk` 抖动，优先值得检查的是：

1. 训练 / 部署两侧的 observation 是否完全一致
2. `action_scale` 是否过大
3. default pose / stiffness / damping 是否完全对齐
4. 是否需要引入 history observation
5. 低层控制出口是否需要更贴近原始 HTWK deploy 语义

---

## 10. 结论

`base_walk` 更容易抖，而 `locomotion` 更稳，最本质的原因不是某一个参数，而是：

> `locomotion` 是一套原生的、历史观测驱动的、全身控制 walking 任务；  
> `base_walk` 则是一个单帧、腿部主导、显式步态相位驱动的任务，对时序误差、噪声和低层控制语义更敏感。

因此，即使把姿态、PD、动作缩放这些表面参数对齐了，`base_walk` 依然可能比 `locomotion` 更容易抖动。

