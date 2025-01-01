import argparse
import os
import pickle
import shutil
import time
from collections import deque

import torch

from panda_env import PandaEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 8,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 100,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 7,  # 7 joints for Panda
        "default_joint_angles": {  # [rad] from panda_nohand.xml keyframe
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": -1.57079,
            "joint5": 0.0,
            "joint6": 1.57079,
            "joint7": -0.7853,
        },
        "dof_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ],
        # PD control parameters from control_your_robot.py
        "kp": 4500.0,  # Position gain
        "kd": 450.0,   # Velocity gain
        # termination conditions
        # "termination_if_roll_greater_than": 90,   # degree
        # "termination_if_pitch_greater_than": 90,  # degree
        "termination_if_reach_threshold": 0.1,  # 目標位置に50cm以内に近づいたら成功
        # base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],  # w, x, y, z
        "episode_length_s": 10.0,
        "resampling_time_s": 10.0,
        "action_scale": 1.0,
        "simulate_action_latency": True,
        "clip_actions": 3.14,
    }

    obs_cfg = {
        "num_obs": 31,  # 3 (target_pos) + 7 (dof_pos) + 7 (dof_vel) + 7 (last_actions) + 7 (actions)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "target_pos": 1.0,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "reward_scales": {
            "reaching_pose": 1.0,          # エンドエフェクタの位置追従
            "time_efficiency": 0.5,        # 時間効率の重み
            # "action_rate": -0.1,           # アクション変化の抑制
            # "action_regulation": -0.01,     # アクション大きさの抑制
            # "joint_acc": -0.1,             # 関節加速度の抑制
            # "joint_limit": -0.5,           # 関節限界への接近抑制
        },
    }

    command_cfg = {
        "num_commands": 3,  # x, y, z position of target
        "pos_range": [  # Target position ranges
            [0.3, 0.7],   # x
            [-0.5, 0.5],   # y
            [0.2, 1.0],    # z
        ],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="panda-reaching")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    env = PandaEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        # show_viewer=True,
        show_viewer=False,

    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    
    # 設定の保存
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/arm/panda_train.py
"""
