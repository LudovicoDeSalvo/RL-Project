from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch

from dqn_agent import DQNAgent


def make_env(env_name: str, video_folder: str, name_prefix: str) -> gym.Env:
    try:
        env = gym.make(env_name, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_name)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda _: True,
    )
    return env


def run_video(checkpoint: str, env_name: str, episodes: int, seed: int | None, out_name: str, video_dir: str) -> None:
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    env = make_env(env_name, video_dir, out_name)
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, epsilon_start=0.0, epsilon_end=0.0)
    agent.policy_net.load_state_dict(torch.load(checkpoint, map_location=agent.device))
    agent.policy_net.eval()

    for _ in range(episodes):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        while not done:
            action = agent.select_action(state)
            step_out = env.step(action)
            if len(step_out) == 5:
                state, _, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, _, done, _ = step_out

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Registra un video di una policy salvata.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="policy_run")
    parser.add_argument("--video-dir", default="videos")
    args = parser.parse_args()

    run_video(args.checkpoint, args.env, args.episodes, args.seed, args.out, args.video_dir)


if __name__ == "__main__":
    main()
