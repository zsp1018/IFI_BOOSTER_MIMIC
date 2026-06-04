# Booster Deploy

Booster Deploy 是一个轻量级部署框架，支持在 Booster 机器人上运行控制策略（sim2real），也支持在 MuJoCo（sim2sim）和 Webots（内部 sim2sim）中运行控制策略。该系统采用了许多 IsaacLab 中成熟的设计，提供模块化抽象，使策略能够在仿真平台和真实机器人平台上统一执行。


## 前置条件

| 环境 | 说明 |
|-------------|-------|
| Booster firmware >= v1.4 | 真实机器人部署所必需。 |
| Python 3.10+ | 机器人上已安装。 |
| ROS 2 Humble | `/low_state` + `/joint_ctrl` 话题所必需。机器人上已安装。 |
| MuJoCo / Webots | 可选；如果计划运行对应的仿真器，请安装。 |


## 运行部署

### 添加并列出任务：
   1. 在 `tasks/` 下为你的任务创建一个子文件夹。
   2. 实现 `Policy`/`PolicyCfg`，并提供一个引用该策略的 `ControllerCfg`。
   3. 将策略检查点放在 `models/` 下，并在配置中引用对应路径。
   4. 在任务注册表中注册你的 `ControllerCfg` 配置（注册方式可参考已有任务）。
   5. 查看所有可用任务：
      ```bash
      python3 scripts/deploy.py --list
      ```

### 运行 Sim2Sim（MuJoCo）

- 下载并安装 BoosterAssets：
   - 克隆 [booster_assets](https://github.com/BoosterRobotics/booster_assets)，其中包含 Booster 机器人模型和相关资源。
   - 按照该仓库中的说明安装 booster_assets Python 辅助工具。

- 在本地机器上安装 Python 依赖：
   ```
   pip install -r requirements.txt
   ```

- 在 MuJoCo 中启动任务：
   ```bash
   python scripts/deploy.py --task <TASK_NAME> --mujoco
   ```

### 运行 Sim2Real（真实机器人）

**重要**：在继续操作之前，请确保机器人上已安装 [Booster Firmware](https://booster.feishu.cn/wiki/E3q5wF5SnitXZgkY18Uc8odBnXb) >= v1.4。

**注意**：如果你计划部署到 T1 标准版机器人，需要选择部署到**运动控制板**，而不是感知板。

- 在本地通过 Sim2Sim 完成任务测试后，将项目复制到机器人上。

- 在机器人上安装 Booster Robotic SDK：
   - 将最新的 [Booster Robotics SDK](https://github.com/BoosterRobotics/booster_robotics_sdk) 仓库克隆到机器人上。
   - 按照 SDK 仓库中的构建说明进行操作。
   - **重要**：请确保构建并安装 Python 绑定：
     ```bash
     cd booster_robotics_sdk
     mkdir build && cd build
     cmake .. -DBUILD_PYTHON_BINDING=ON
     make -j$(nproc)
     sudo make install
     ```

- 在机器人上安装 Python 依赖：
   ```
   pip install -r requirements.txt
   ```

- SSH 登录机器人，并通过 source 提供的 setup 脚本启动 ROS 2 环境：
   ```bash
   source /opt/booster/BoosterRos2Interface/install/setup.bash
   ```

- 在机器人上启动任务，并按照命令行中显示的提示操作。
   ```bash
   python3 scripts/deploy.py --task <TASK_NAME>
   ```


## 仓库结构

```
booster_deploy/
├─ booster_deploy/           # 控制器、策略、工具
├─ scripts/                  # 入口脚本（deploy.py）
├─ tasks/                    # 任务注册表和配置
├─ requirements.txt          # Python 依赖
└─ fastdds_profile.xml       # ROS 2 的默认 FastDDS 设置
```

关键模块：
- `booster_deploy/`：核心模块，为仿真器和实体机器人提供统一抽象，并通过 ROS 2 处理通信（实现 `/low_state` 订阅者和 `/low_cmd` 发布者，用于将策略桥接到硬件）。
- `booster_deploy/robots/`：机器人配置模块。该文件夹包含 Booster 机器人配置，通过定义 `RobotCfg` 描述：
    - 关节名称和机身名称
    - 默认关节位置
    - 默认关节刚度（`joint_stiffness`）和阻尼（`joint_damping`）
    - 力矩限制
    - 用于 MuJoCo 模型加载的 `mjcf_path`
    - `prepare_state`（进入自定义模式时使用的准备姿态、刚度和阻尼）

 - `tasks/`：用户任务定义和实现。每个任务模块包含：
    - 实现推理逻辑的 `Policy`/`PolicyCfg` 类；
    - 描述任务配置（包括策略）的 `ControllerCfg` 类；
    - 使用 `ControllerCfg` 实例注册任务。

   典型任务布局（示例）：

   ```text
   tasks/my_task/
   ├─ __init__.py        # 通过 utils.register.register_task 注册任务
   ├─ task.py            # Policy 和 ControllerCfg 实现
   ├─ models/            # 可选的策略检查点
   └─ motions/           # 可选的动作原语或记录
   ```
