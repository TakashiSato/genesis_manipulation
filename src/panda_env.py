import torch
import math
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
import numpy as np


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class PandaEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        is_eval=False,
        device="cuda",
    ):
        self.device = torch.device(device)
        self.show_viewer = show_viewer
        self.is_eval = is_eval

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=self.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # add target pose sphere
        if self.show_viewer:
            self.target_sphere = self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=(0.5, 0.0, 0.2),
                    radius=0.01,
                    visualization=True,
                    collision=False,
                    fixed=True,
                )
            )

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")
            # pos=self.base_init_pos.cpu().numpy(),
            # quat=self.base_init_quat.cpu().numpy(),
            # ),
        )

        # 障害物の設定
        self.num_obstacles = self.env_cfg.get("num_obstacles", 3)  # 障害物の数
        self.obstacle_radius = self.env_cfg.get("obstacle_radius", 0.05)  # 障害物の半径
        self.obstacle_margin = self.env_cfg.get("obstacle_margin", 0.1)
        
        # 障害物の位置を保存するバッファ
        self.obstacle_positions = torch.zeros(
            (self.num_envs, self.num_obstacles, 3),
            device=self.device,
            dtype=gs.tc_float
        )
        
        # 障害物の可視化用のエンティティを作成
        self.obstacles = []
        for _ in range(self.num_obstacles):
            obstacle = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    radius=self.obstacle_radius,
                    visualization=True,
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Default(
                    color=(0.0, 1.0, 0.0, 0.5),  # 緑色で半透明
                ),
            )
            self.obstacles.append(obstacle)

        # 監視するリンク名のリスト
        self.monitored_links = [
            "link1",  # 第1関節
            "link2",  # 第2関節
            "link3",  # 第3関節
            "link4",  # 第4関節
            "link5",  # 第5関節
            "link6",  # 第6関節
            "link7",  # 第7関節
            "hand",   # エンドエフェクタ
        ]
        
        # 各リンクのEntity
        self.link_entities = {
            link_name: self.robot.get_link(link_name)
            for link_name in self.monitored_links
        }
        self.obstacle_distances = torch.zeros(
            (self.num_envs, len(self.monitored_links), self.num_obstacles),
            device=self.device,
            dtype=gs.tc_float
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg["dof_names"]
        ]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )

        # initialize buffers
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.commands = torch.zeros(
            (self.num_envs, 3),  # [num_envs, 3] の形状（xyz座標）
            device=self.device,
            dtype=gs.tc_float,
        )
        self.commands_scale = torch.tensor(
            [
                pos_range[1] - pos_range[0]
                for pos_range in self.command_cfg["pos_range"]
            ],  # 各軸の範囲の大きさをスケールとして使用
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.default_dof_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["dof_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        """目標位置をリサンプリング"""
        # 各環境に対して3次元（xyz）の目標位置を生成
        for i in range(3):  # x, y, z座標それぞれに対して
            pos_range = self.command_cfg["pos_range"][i]
            self.commands[envs_idx, i] = (
                torch.rand(len(envs_idx), device=self.device)
                * (pos_range[1] - pos_range[0])
                + pos_range[0]
            )

        # self.commands[envs_idx, 0] = 0.5
        # self.commands[envs_idx, 1] = 0.0
        # self.commands[envs_idx, 2] = 0.8

    def _resample_obstacles(self, envs_idx):
        """障害物の位置をリサンプリング"""
        for i in range(self.num_obstacles):
            # ワークスペース内でランダムな位置を生成
            for j in range(3):  # x, y, z座標
                pos_range = self.command_cfg["pos_range"][j]
                self.obstacle_positions[envs_idx, i, j] = (
                    torch.rand(len(envs_idx), device=self.device)
                    * (pos_range[1] - pos_range[0])
                    + pos_range[0]
                )
            
            # 障害物の位置を更新
            self.obstacles[i].set_pos(self.obstacle_positions[:, i])

    def step(self, actions):
        # 観測値の各要素をプリント
        # print("Observation components:")
        # print("- commands:", self.obs_buf[:, :3])  # 最初の3次元
        # print("- dof_pos:", self.obs_buf[:, 3:10]) 
        # print("- dof_vel:", self.obs_buf[:, 10:17])
        # print("- last_actions:", self.obs_buf[:, 17:24])
        # print("- actions:", self.obs_buf[:, 24:31])

        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = (
            exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        )
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )

        ee_link = self.robot.get_link("hand")
        self.ee_pos = ee_link.get_pos()
        self.pos_error = torch.norm(self.commands - self.ee_pos, dim=1)  # ユークリッド距離

        # 障害物との距離を計算
        for i, link in enumerate(self.link_entities.values()):
            link_pos = link.get_pos()
            link_pos_expanded = link_pos.unsqueeze(1)
            distances = torch.norm(link_pos_expanded - self.obstacle_positions, dim=2)
            self.obstacle_distances[:, i] = distances

        # 全リンクと障害物との最小距離を計算
        min_distances_per_link = torch.min(self.obstacle_distances, dim=2)[0]  # [num_envs, num_links]
        global_min_distance = torch.min(min_distances_per_link, dim=1)[0]  # [num_envs]

        if self.show_viewer:
            # 全障害物に対する各リンクとの最小距離を計算
            min_distances_per_obstacle = torch.min(self.obstacle_distances, dim=1)[0]  # [num_envs, num_obstacles]

            # 干渉している障害物の色を変える(env0のみ)
            is_colliding_obstacles = min_distances_per_obstacle[0] < (self.obstacle_radius + self.obstacle_margin)
            for i in range(self.num_obstacles):
                if is_colliding_obstacles[i]:
                    self.scene.draw_debug_spheres(poss=self.obstacle_positions[0, i], radius=0.05, color=(1, 0, 0, 0.5))


        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # check termination and reset
        # self.max_episode_length = 1000000000000
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # print(f"reset_buf: {self.reset_buf} episode_length_buf: {self.episode_length_buf}")

        # 目標位置への到達判定による終了条件
        reached = self.pos_error < self.env_cfg["termination_if_reach_threshold"]
        # print(f"commands: {self.commands}")
        # print(f"pos_error: {self.pos_error}, reached: {reached}")
        self.reset_buf |= reached

        # # 障害物との干渉による終了条件(train時のみ)
        # if not self.is_eval:
        #     is_colliding = global_min_distance < (self.obstacle_radius + self.obstacle_margin)
        #     self.reset_buf |= is_colliding

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        reset_indices = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(reset_indices)

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        # # print(f"commands shape {self.commands.shape}, dof_pos shape {self.dof_pos.shape}, dof_vel shape {self.dof_vel.shape}, last_actions shape {self.last_actions.shape}, actions shape {self.actions.shape}, obstacle_positions shape {self.obstacle_positions.shape}, reshaped: {self.obstacle_positions.reshape(self.num_envs, -1).shape}")
        # import sys
        # sys.exit(1)
        self.obs_buf = torch.cat(
            [
                self.commands,                    # 3 (target position)
                self.obstacle_positions.reshape(self.num_envs, -1),  # 3 * num_obstacles
                self.dof_pos,                     # 7 (joint positions)
                self.dof_vel,                     # 7 (joint velocities)
                self.last_actions,                # 7 (previous actions)
                self.actions,                     # 7 (current actions)
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # 現在の状態をプリント
        # print("Current state:")
        # print("Target position:", self.commands[0])  # 目標位置
        # print("Current EE position:", self.ee_pos[0])  # 現在の手先位置
        # print("Position error:", self.pos_error[0])  # 位置誤差
        # print("Generated action:", actions[0])  # 生成されたアクション

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        # self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        # self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

        # 球体の位置を更新（表示されている環境の目標位置を使用）
        if self.show_viewer:
            self.target_sphere.set_pos(self.commands.cpu().numpy())

        # 障害物の位置をリセット
        self._resample_obstacles(envs_idx)

        # デバッグオブジェクトをクリア
        if self.show_viewer and (0 in envs_idx):
            self.scene.clear_debug_objects()

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_reaching_pose(self):
        """位置への到達を報酬化"""
        pos_error_xyz = torch.abs(self.commands - self.ee_pos)
        reward_xyz = -pos_error_xyz

        return torch.sum(reward_xyz, dim=1)

    def _reward_time_efficiency(self):
        """時間効率を報酬化（ペナルティベース）"""
        pos_error_xyz = torch.abs(self.commands - self.ee_pos)
        success_mask = torch.all(pos_error_xyz < self.env_cfg["termination_if_reach_threshold"], dim=1)
        
        # 時間に基づくペナルティ（-1.0 → 0.0）
        time_penalty = -(self.episode_length_buf / self.max_episode_length)
        
        # 成功時のみペナルティを考慮
        return torch.where(
            success_mask,
            time_penalty,
            torch.ones_like(time_penalty) * -1.0  # 未到達時は最大ペナルティ
        )

    def _reward_action_rate(self):
        """アクションの急激な変化を抑制"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_joint_acc(self):
        """関節加速度を抑制（スケーリング調整版）"""
        joint_acc = (self.dof_vel - self.last_dof_vel) / self.dt
        
        # 各関節の加速度を個別に正規化
        normalized_acc = joint_acc / (2.0 * self.env_cfg["clip_actions"] / (self.dt * self.dt))
        
        # 二乗和の平均を取ることでスケーリング
        return -torch.mean(torch.square(normalized_acc), dim=1)

    def _reward_action_regulation(self):
        """アクションの大きさを抑制"""
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_joint_limit(self):
        """関節限界への接近を抑制"""
        # 各関節の可動範囲を定義
        joint_ranges = torch.tensor(
            [
                [-2.8973, 2.8973],  # joint1
                [-1.7628, 1.7628],  # joint2
                [-2.8973, 2.8973],  # joint3
                [-3.0718, -0.0698],  # joint4
                [-2.8973, 2.8973],  # joint5
                [-0.0175, 3.7525],  # joint6
                [-2.8973, 2.8973],  # joint7
            ],
            device=self.device,
        )

        # 各関節の可動範囲の中心を計算
        joint_centers = (joint_ranges[:, 1] + joint_ranges[:, 0]) / 2

        # 現在の関節角度と中心位置との差を計算
        joint_deviations = torch.abs(self.dof_pos - joint_centers)

        # 可動範囲の半分の大きさで正規化
        joint_ranges_half = (joint_ranges[:, 1] - joint_ranges[:, 0]) / 2
        normalized_deviations = joint_deviations / joint_ranges_half

        # ペナルティを計算（関節限界に近づくほど大きくなる）
        return torch.sum(torch.square(normalized_deviations), dim=1)

    def _reward_obstacle_avoidance(self):
        """障害物回避の報酬（距離に基づくペナルティ）"""
        collision_threshold = self.obstacle_radius
        margin_threshold = self.obstacle_radius + self.obstacle_margin

        # 距離がmargin_thresholdより遠い場合はペナルティなし
        penalty = torch.zeros_like(self.obstacle_distances)

        # 距離がcollision_threshold以下の場合、ペナルティを-1から0の範囲に線形に設定
        collision_penalty = torch.where(
            self.obstacle_distances < collision_threshold,
            -1.0 + (self.obstacle_distances / collision_threshold),
            penalty
        )

        # 距離がmargin_thresholdより近く、collision_thresholdより遠い場合は徐々にペナルティを増加
        distance_penalty = torch.where(
            (self.obstacle_distances >= collision_threshold) & (self.obstacle_distances < margin_threshold),
            1.0 / (self.obstacle_distances - collision_threshold + 1e-6),  # 0除算を避けるために小さな値を加算
            penalty
        )

        # 各リンクと障害物の距離のペナルティを合計
        total_penalty = torch.sum(torch.sum(distance_penalty, dim=1), dim=1) + torch.sum(collision_penalty, dim=(1, 2))

        # リンク数と障害物数で正規化
        avg_penalty = total_penalty / (len(self.monitored_links) * self.obstacle_positions.size(1))

        return -avg_penalty  # ペナルティなので負の値