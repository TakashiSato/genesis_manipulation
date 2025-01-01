import argparse
import os
import pickle
import time

import numpy as np
import torch
from panda_env import PandaEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="panda-reaching")
    parser.add_argument("--ckpt", type=int, default=1000)
    parser.add_argument("--resampling_time", type=float, default=3.0)  # 目標位置の再サンプリング時間[s]
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    
    # 評価時は報酬計算を無効化（オプション）
    # reward_cfg["reward_scales"] = {}

    env = PandaEnv(
        num_envs=1,  # 評価時は1環境
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,  # ビューワーを表示
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()

    with torch.no_grad():
        while True:  # 無限ループで実行
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/arm/panda_eval.py -e panda-reaching --ckpt 1000 --resampling_time 2.0
"""
