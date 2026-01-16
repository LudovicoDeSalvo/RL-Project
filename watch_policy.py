from __future__ import annotations

import argparse

import gymnasium as gym
import torch

from dqn_agent import DQNAgent


def make_env(env_name: str) -> gym.Env:
    try:
        return gym.make(env_name, render_mode="human")
    except TypeError:
        return gym.make(env_name)


def run_policy(checkpoint: str, env_name: str, episodes: int, seed: int | None) -> None:
    env = make_env(env_name)
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
            if getattr(env, "render_mode", None) is None:
                env.render()

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizza una policy addestrata su CartPole.")
    parser.add_argument("--checkpoint", default="dqn_policy_run1.pt")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_policy(args.checkpoint, args.env, args.episodes, args.seed)


if __name__ == "__main__":
    main()
