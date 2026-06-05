import argparse
from pathlib import Path
import sys

# 允许脚本从仓库根目录直接导入本地包，例如 tasks、booster_deploy 等。
# 运行方式通常是：
#   python scripts/deploy.py --task <task_name> ...
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# 命令行参数解析器。这个脚本是部署入口，可以列出任务、启动 MuJoCo
# 仿真、启动 Webots 仿真，或者连接真实机器人运行策略。
parser = argparse.ArgumentParser()

# --task 和 --list 二选一：
# - --task 指定要运行的任务配置，例如 parameter_walk
# - --list 只打印当前已注册的任务，不启动控制器
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--task", type=str, help="Name of the configuration file.")
group.add_argument("-l", "--list", action="store_true", dest="list_tasks",
                   default=False, help="list available tasks")

# SDK 通信使用的网络接口或地址。真机部署时会传给
# ChannelFactory.Instance().Init(0, args.net)。
parser.add_argument("--net", type=str, default="127.0.0.1",
                    help="Network interface for SDK communication.")

# 使用 MuJoCo 控制器运行。开启后不会初始化真实机器人 SDK。
parser.add_argument("--mujoco", action="store_true", default=False,
                    help="deploy in mujoco simulation")

# 使用 Webots 仿真。代码路径仍然走 BoosterRobotPortal，
# 但会启用仿真时间，并对部分关节阻尼做 Webots 适配。
parser.add_argument("--webots", action="store_true", default=False,
                    help="deploy in webots simulation")

# 策略网络推理设备。部署真机通常用 cpu；如果本机有 GPU，
# 仿真评估时也可以传 --device cuda。
parser.add_argument(
    "--device", type=str, default="cpu",
    help="Device to run the evaluation on (e.g., 'cpu', 'cuda')")
args = parser.parse_args()


def main():
    # 导入 tasks 包并递归扫描其子模块。
    # 每个 task 模块在被 import 时通常会执行 register_task(...)，
    # 把自己的配置类注册到全局 registry 里。这里先全部 import，
    # 后面 get_task(args.task) 才能按名字找到对应配置。
    import pkgutil
    import tasks as tasks_pkg

    # 自动导入 tasks 下所有子模块，例如 tasks.parameter_walk。
    # prefix="tasks." 确保 import 使用完整包名，避免相对路径歧义。
    for mod_info in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
        full_name = mod_info.name
        try:
            __import__(full_name)
        except Exception as e:
            # 如果某个任务导入失败，直接抛出原始异常，方便看到真实报错。
            raise e
    from booster_deploy.utils.registry import get_task, list_tasks

    # 只列出任务时，不创建控制器，也不加载 SDK。
    # 输出内容包含任务名和对应配置类的完整 Python 路径。
    if args.list_tasks:
        print("Available tasks:")
        for task_name, cfg in list_tasks().items():
            cls = type(cfg)
            full_cls = f"{cls.__module__}.{cls.__qualname__}"
            print(f"  {task_name}\t:\t{full_cls}")
        sys.exit(0)

    # 根据 --task 名字从 registry 中取出任务配置对象。
    # 例如 parameter_walk 会返回该任务的配置实例，里面包含：
    # - policy: 策略模型路径、观测缩放、推理设备等
    # - robot: 关节 PD、关节限位、默认姿态等
    # - controller/task: 控制循环和任务自定义参数
    try:
        task_cfg = get_task(args.task)
    except KeyError:
        print(f"Unknown task '{args.task}'. Available tasks: {list(list_tasks().keys())}")
        sys.exit(1)

    # 设置策略网络推理设备。这里覆盖任务配置里的默认值，
    # 因此命令行 --device 优先级更高。
    task_cfg.policy.device = args.device

    # 根据运行模式选择控制器：
    # - --mujoco: 走 MuJoCoController，不需要真实机器人 SDK
    # - 默认/--webots: 走 BoosterRobotPortal，需要 SDK 包；
    #   Webots 只是使用仿真时间并调整部分参数
    if args.mujoco:
        # MuJoCo 模式：直接创建 MuJoCo 控制器并进入控制循环。
        from booster_deploy.controllers.mujoco_controller import MujocoController

        MujocoController(task_cfg).run()
    else:
        # 真机或 Webots 模式：先初始化 Booster SDK 通信通道。
        # args.net 决定 SDK 绑定/连接的网络接口；默认 127.0.0.1
        # 更适合本机仿真，真机部署时通常要换成机器人通信网卡地址。
        try:
            from booster_robotics_sdk_python import ChannelFactory  # type: ignore
            ChannelFactory.Instance().Init(0, args.net)
        except ImportError as e:
            # 没装 SDK 时无法走真机/Webots 入口；如果只是想仿真，
            # 可以改用 --mujoco，它不会导入 booster_robotics_sdk_python。
            print(
                "Error: booster_robotics_sdk_python is not installed.\n"
                "Please install it to use real robot deployment.\n"
                "For MuJoCo simulation, use --mujoco flag instead."
            )
            sys.exit(1)

        # Webots 模式下脚踝关节阻尼需要单独调小。
        # 这里用负索引定位末尾的几个踝关节：
        # -8, -7, -2, -1 分别对应配置中靠后的踝关节索引。
        # 注意：这会直接修改 task_cfg.robot.joint_damping。
        if args.webots:
            ankles = [-8, -7, -2, -1]  # indices of ankle joints
            for i in ankles:
                task_cfg.robot.joint_damping[i] = 0.5

        # BoosterRobotPortal 封装真实机器人/SDK 控制循环：
        # - use_sim_time=False: 真机时间
        # - use_sim_time=True: Webots 仿真时间
        # 使用 context manager 可以保证退出时释放通信资源。
        from booster_deploy.controllers.booster_robot_controller import BoosterRobotPortal
        with BoosterRobotPortal(task_cfg, use_sim_time=args.webots) as portal:
            portal.run()


# Python 标准入口保护：只有直接运行本文件时才启动部署流程；
# 如果被其他模块 import，不会立刻执行 main()。
if __name__ == "__main__":
    main()
