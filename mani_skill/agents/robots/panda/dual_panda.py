from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
from sapien.core import Pose
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

from mani_skill.sensors.camera import CameraConfig


@register_agent()
class Dual_Panda(BaseAgent):
    uid = "dual_panda"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/dual_panda.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            panda_leftfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            panda_rightfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            # TODO: Set the rest pose and figure out the order of the joints
            qpos=np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0, # base_revolute_z
                    0.0, # panda_joint1_1
                    1.0, # camera_link_joint
                    0.0, # panda_joint1_2
                    1.0, # camera_link_joint_2
                    np.pi / 8, # panda_joint2_1
                    np.pi / 8, # panda_joint2_2
                    0.0, # panda_joint3_1
                    0.0, # panda_joint3_2
                    -np.pi * 5 / 8, # panda_joint4_1
                    -np.pi * 5 / 8, # panda_joint4_2
                    0.0, # panda_joint5_1
                    0.0, # panda_joint5_2
                    np.pi * 3 / 4, # panda_joint6_1
                    np.pi * 3 / 4, # panda_joint6_2
                    np.pi / 4, # panda_joint7_1
                    np.pi / 4, # panda_joint7_2
                    0.04, # panda_finger_joint1_1
                    0.04, # panda_finger_joint2_1
                    0.04, # panda_finger_joint1_2
                    0.04, # panda_finger_joint2_2
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "base_prismatic_x",
        "base_prismatic_y",
        "base_prismatic_z",
        "base_revolute_z",
        "panda_joint1_1",
        "panda_joint2_1",
        "panda_joint3_1",
        "panda_joint4_1",
        "panda_joint5_1",
        "panda_joint6_1",
        "panda_joint7_1",
        "panda_joint1_2",
        "panda_joint2_2",
        "panda_joint3_2",
        "panda_joint4_2",
        "panda_joint5_2",
        "panda_joint6_2",
        "panda_joint7_2",
    ]
    gripper_joint_names = [
        "panda_finger_joint1_1",
        "panda_finger_joint2_1",
        "panda_finger_joint1_2",
        "panda_finger_joint2_2",
    ]
    ee_link_name = ["panda_hand_tcp_1", "panda_hand_tcp_2"]

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        # NOTE: The ee controller may not work because the ee_link is tuple which is not supported
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.finger1_link_1 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger_1"
        )
        self.finger2_link_1 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger_1"
        )
        self.finger1pad_link_1 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger_pad_1"
        )
        self.finger2pad_link_1 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger_pad_1"
        )
        self.tcp_1 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name[0]
        )

        self.finger1_link_2 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger_2"
        )
        self.finger2_link_2 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger_2"
        )
        self.finger1pad_link_2 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger_pad_2"
        )
        self.finger2pad_link_2 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger_pad_2"
        )
        self.tcp_2 = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name[1]
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces_1 = self.scene.get_pairwise_contact_forces(
            self.finger1_link_1, object
        )
        r_contact_forces_1 = self.scene.get_pairwise_contact_forces(
            self.finger2_link_1, object
        )
        l_contact_forces_2 = self.scene.get_pairwise_contact_forces(
            self.finger1_link_2, object
        )
        r_contact_forces_2 = self.scene.get_pairwise_contact_forces(
            self.finger2_link_2, object
        )
        lforce_1 = torch.linalg.norm(l_contact_forces_1, axis=1)
        rforce_1 = torch.linalg.norm(r_contact_forces_1, axis=1)
        lforce_2 = torch.linalg.norm(l_contact_forces_2, axis=1)
        rforce_2 = torch.linalg.norm(r_contact_forces_2, axis=1)

        # direction to open the gripper
        ldirection_1 = self.finger1_link_1.pose.to_transformation_matrix()[..., :3, 1]
        rdirection_1 = -self.finger2_link_1.pose.to_transformation_matrix()[..., :3, 1]
        ldirection_2 = self.finger1_link_2.pose.to_transformation_matrix()[..., :3, 1]
        rdirection_2 = -self.finger2_link_2.pose.to_transformation_matrix()[..., :3, 1]
        langle_1 = common.compute_angle_between(ldirection_1, l_contact_forces_1)
        rangle_1 = common.compute_angle_between(rdirection_1, r_contact_forces_1)
        langle_2 = common.compute_angle_between(ldirection_2, l_contact_forces_2)
        rangle_2 = common.compute_angle_between(rdirection_2, r_contact_forces_2)
        lflag_1 = torch.logical_and(
            lforce_1 >= min_force, torch.rad2deg(langle_1) <= max_angle
        )
        rflag_1 = torch.logical_and(
            rforce_1 >= min_force, torch.rad2deg(rangle_1) <= max_angle
        )
        lflag_2 = torch.logical_and(
            lforce_2 >= min_force, torch.rad2deg(langle_2) <= max_angle
        )
        rflag_2 = torch.logical_and(
            rforce_2 >= min_force, torch.rad2deg(rangle_2) <= max_angle
        )
        return torch.logical_and(torch.logical_and(lflag_1, rflag_1), torch.logical_and(lflag_2, rflag_2))

    def is_static(self, threshold: float = 0.2):
        # TODO: Find the index of the gripper joints
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="panda_camera_front",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # Looking forward
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_front"],
            ),
            CameraConfig(
                uid="panda_camera_back",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_back"],
            ),
            CameraConfig(
                uid="panda_camera_left",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # Looking forward
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_left"],
            ),
            CameraConfig(
                uid="panda_camera_right",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # Looking forward
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_right"],
            ),
            CameraConfig(
                uid="panda_camera_top",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # Looking forward
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_top"],
            ),
            CameraConfig(
                uid="panda_camera_front_2",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # Looking forward
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_front_2"],
            ),
            CameraConfig(
                uid="panda_camera_back_2",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_back_2"],
            ),
            CameraConfig(
                uid="panda_camera_left_2",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # Looking forward
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_left_2"],
            ),
            CameraConfig(
                uid="panda_camera_right_2",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # Looking forward
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_right_2"],
            ),
            CameraConfig(
                uid="panda_camera_top_2",
                pose=Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # Looking forward
                width=512,
                height=512,
                fov=np.pi * 2 / 3,  # 120 degrees FOV
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link_top_2"],
            ),
            
        ]
