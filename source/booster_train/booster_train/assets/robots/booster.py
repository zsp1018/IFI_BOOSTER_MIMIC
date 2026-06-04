import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import *
from booster_train.assets.robots import actuator
from booster_train.assets.robots.actuator import (
    BoosterDelayedImplicitActuatorCfg,
    BoosterDelayedPDActuatorCfg,
    DelayedImplicitActuatorCfg
)

from booster_assets import BOOSTER_ASSETS_DIR

BOOSTER_K1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        joint_pos={
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,

    actuators={
        "legs": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            booster_joint_cfgs={
                ".*_Hip_Pitch": actuator.BoosterJointE6408(natural_freq = 4.0, damping_ratio = 1.5),
                ".*_Hip_Roll": actuator.BoosterJointE4315(natural_freq = 4.0, damping_ratio = 1.5),
                ".*_Hip_Yaw": actuator.BoosterJointE4310(natural_freq = 4.0, damping_ratio = 1.5),
                ".*_Knee_Pitch": actuator.BoosterJointE6416(natural_freq = 4.0, damping_ratio = 1.0),
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Ankle_Pitch",
                ".*_Ankle_Roll",
            ],
            booster_joint_cfgs={
                ".*_Ankle_Pitch": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=0,
                    natural_freq = 4.0,
                    damping_ratio = 1.5,
                ),
                ".*_Ankle_Roll": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=1,
                    natural_freq = 4.0,
                    damping_ratio = 1.5,
                ),
            },
        ),
        "arms": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            booster_joint_cfgs=actuator.BoosterJointR14(),
        ),
        "head": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*Head.*"],
            booster_joint_cfgs=actuator.BoosterJointHT4438(),
        ),
    }
)

K1_ACTION_SCALE = {}
for a in BOOSTER_K1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            K1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

print(f'{BOOSTER_K1_CFG.actuators=}')
print(f'{K1_ACTION_SCALE=}')


BOOSTER_T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.70),
        joint_pos={
            ".*_Shoulder_Pitch": 0.2,
            "Left_Shoulder_Roll": -1.3,
            "Right_Shoulder_Roll": 1.3,
            "Left_Elbow_Yaw": -0.5,
            "Right_Elbow_Yaw": 0.5,
            ".*_Hip_Pitch": -0.2,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            booster_joint_cfgs=actuator.BoosterJointE4310(
                effort_limit=36.0,
            ),
        ),
        "waist": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=["Waist"],
            booster_joint_cfgs=actuator.BoosterJointE6408(
                effort_limit=40.0,
            ),
        ),
        "legs": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            booster_joint_cfgs={
                ".*_Hip_Pitch": actuator.BoosterJointE8112(
                    effort_limit=55.0,
                ),
                ".*_Hip_Roll": actuator.BoosterJointE6408(
                    effort_limit=40.0,
                ),
                ".*_Hip_Yaw": actuator.BoosterJointE6408(
                    effort_limit=40.0,
                ),
                ".*_Knee_Pitch": actuator.BoosterJointE8116(
                    effort_limit=65.0,
                ),
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[
                ".*_Ankle_Pitch",
                ".*_Ankle_Roll",
            ],
            booster_joint_cfgs={
                ".*_Ankle_Pitch": actuator.BoosterT1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4315(
                        effort_limit=50.0,
                    ),
                    serial_index=0,
                ),
                ".*_Ankle_Roll": actuator.BoosterT1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4315(
                        effort_limit=50.0,
                    ),
                    serial_index=1,
                ),
            },
        ),
        "head": BoosterDelayedPDActuatorCfg(
            max_delay=8,
            min_delay=2,
            joint_names_expr=[".*Head.*"],
            booster_joint_cfgs=actuator.BoosterJointDM4310(),
        ),
    },
)

T1_ACTION_SCALE = {}
for a in BOOSTER_T1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            T1_ACTION_SCALE[n] = 0.5 * e[n] / s[n]

# print(f'{BOOSTER_T1_CFG.actuators=}')
# print(f'{T1_ACTION_SCALE=}')
