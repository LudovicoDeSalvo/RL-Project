from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch

from dqn_agent import DQNAgent, ReplayBuffer
from utils import moving_average, plot_rewards, set_seed


def create_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    return env


def step_env(env: gym.Env, action: int):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, _ = result
        done = terminated or truncated
    else:
        next_state, reward, done, _ = result
    return next_state, reward, done


def warmup_buffer(env: gym.Env, buffer: ReplayBuffer, steps: int, use_rmmert: bool) -> int:
    """Collect random transitions to prefill the buffer before training."""
    collected = 0
    while collected < steps:
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        episode = []
        while not done and collected < steps:
            action = env.action_space.sample()
            next_state, reward, done = step_env(env, action)
            episode.append((state, action, reward, next_state, done))
            state = next_state
            collected += 1
        if episode:
            buffer.add_episode(episode, use_distance=use_rmmert)
    return collected


def run_agent(
    env: gym.Env,
    agent: DQNAgent,
    buffer: ReplayBuffer,
    episodes: int,
    batch_size: int,
    warmup: int,
    use_rmmert: bool,
    log_interval: int = 10,
    window: int = 10,
    on_episode_end: Optional[Callable[[int, float, float, DQNAgent], None]] = None,
    eval_env: Optional[gym.Env] = None,
    eval_interval: int = 0,
    eval_episodes: int = 5,
    on_eval: Optional[Callable[[int, float, DQNAgent], None]] = None,
):
    rewards = []
    for episode in range(episodes):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        episode_transitions = []
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = step_env(env, action)
            episode_transitions.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Optional in-episode updates using existing buffer data (no new push until episode end).
            if len(buffer) >= warmup:
                agent.update(buffer, batch_size, weighted=use_rmmert)

        buffer.add_episode(episode_transitions, use_distance=use_rmmert)

        rewards.append(total_reward)
        avg_window = float(np.mean(rewards[-window:]))
        if on_episode_end is not None:
            on_episode_end(episode + 1, total_reward, avg_window, agent)

        if (episode + 1) % log_interval == 0:
            recent_window = rewards[-log_interval:]
            avg = float(np.mean(recent_window))
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"{'ReMERT ' if use_rmmert else ''}AvgReward(last {log_interval}): {avg:.1f}"
            )
        if eval_env is not None and eval_interval > 0 and (episode + 1) % eval_interval == 0:
            eval_avg = evaluate_policy(eval_env, agent, eval_episodes)
            print(
                f"Eval @ episode {episode + 1} | "
                f"{'ReMERT ' if use_rmmert else ''}AvgReward({eval_episodes} eps): {eval_avg:.1f}"
            )
            if on_eval is not None:
                on_eval(episode + 1, eval_avg, agent)
    return rewards


def evaluate_policy(env: gym.Env, agent: DQNAgent, episodes: int) -> float:
    total_reward = 0.0
    for _ in range(episodes):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.policy_net(state_tensor).argmax(dim=1).item())
            next_state, reward, done = step_env(env, action)
            state = next_state
            total_reward += reward
    return total_reward / float(episodes)


def copy_state_dict(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main():
    env_name = "CartPole-v1"
    episodes = 1000
    batch_size = 64
    warmup_steps = 1_000
    buffer_capacity = 50_000
    n_runs = 3
    base_seed = 42
    run_seeds = [base_seed + 100 * i for i in range(n_runs)]

    all_rewards_dqn = []
    all_rewards_rmmert = []

    for run_idx, run_seed in enumerate(run_seeds, start=1):
        print(f"\n=== Run {run_idx}/{n_runs} (seed {run_seed}) ===")
        set_seed(run_seed)

        env_dqn = create_env(env_name, seed=run_seed)
        env_rmmert = create_env(env_name, seed=run_seed)

        state_dim = env_dqn.observation_space.shape[0]
        action_dim = env_dqn.action_space.n

        dqn_agent = DQNAgent(state_dim, action_dim)
        dqn_buffer = ReplayBuffer(buffer_capacity)
        rmmert_agent = DQNAgent(state_dim, action_dim)
        rmmert_buffer = ReplayBuffer(buffer_capacity)
        window = 10
        target_reward = 500.0
        eval_interval = 50
        eval_episodes = 5

        rmmert_target = {"episode": None, "state": None}
        dqn_at_rmmert_target = {"episode": None, "state": None}
        rmmert_states_by_episode: list[Optional[dict]] = [None] * episodes
        best_eval_rmmert = {"avg": float("-inf"), "episode": None, "state": None}
        best_eval_dqn = {"avg": float("-inf"), "episode": None, "state": None}

        if warmup_steps > 0:
            print(f"Warmup buffer DQN with {warmup_steps} random steps...")
            warmup_buffer(env_dqn, dqn_buffer, warmup_steps, use_rmmert=False)
            print(f"Warmup buffer ReMERT with {warmup_steps} random steps...")
            warmup_buffer(env_rmmert, rmmert_buffer, warmup_steps, use_rmmert=True)

        def on_rmmert_episode(ep: int, total_reward: float, avg: float, agent: DQNAgent) -> None:
            rmmert_states_by_episode[ep - 1] = copy_state_dict(agent.policy_net)
            if rmmert_target["episode"] is None and total_reward >= target_reward:
                rmmert_target["episode"] = ep
                rmmert_target["state"] = rmmert_states_by_episode[ep - 1]

        eval_env_rmmert = create_env(env_name, seed=run_seed + 1)

        def on_rmmert_eval(ep: int, avg: float, agent: DQNAgent) -> None:
            if avg > best_eval_rmmert["avg"]:
                best_eval_rmmert["avg"] = avg
                best_eval_rmmert["episode"] = ep
                best_eval_rmmert["state"] = copy_state_dict(agent.policy_net)

        print("\nTraining DQN + ReMERT (prioritized on distance to end)...")
        rewards_rmmert = run_agent(
            env_rmmert,
            rmmert_agent,
            rmmert_buffer,
            episodes,
            batch_size,
            warmup_steps,
            use_rmmert=True,
            log_interval=10,
            window=window,
            on_episode_end=on_rmmert_episode,
            eval_env=eval_env_rmmert,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            on_eval=on_rmmert_eval,
        )

        rewards_rmmert_arr = np.array(rewards_rmmert, dtype=np.float32)
        rmmert_ma = moving_average(rewards_rmmert_arr, window)

        eval_env_dqn = create_env(env_name, seed=run_seed + 2)

        def on_dqn_eval(ep: int, avg: float, agent: DQNAgent) -> None:
            if avg > best_eval_dqn["avg"]:
                best_eval_dqn["avg"] = avg
                best_eval_dqn["episode"] = ep
                best_eval_dqn["state"] = copy_state_dict(agent.policy_net)

        def on_dqn_episode(ep: int, _: float, avg: float, agent: DQNAgent) -> None:
            if rmmert_target["episode"] is not None and ep == rmmert_target["episode"]:
                dqn_at_rmmert_target["episode"] = ep
                dqn_at_rmmert_target["state"] = copy_state_dict(agent.policy_net)

        print("Training standard DQN...")
        rewards_dqn = run_agent(
            env_dqn,
            dqn_agent,
            dqn_buffer,
            episodes,
            batch_size,
            warmup_steps,
            use_rmmert=False,
            log_interval=10,
            window=window,
            on_episode_end=on_dqn_episode,
            eval_env=eval_env_dqn,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            on_eval=on_dqn_eval,
        )

        rewards_dqn_arr = np.array(rewards_dqn, dtype=np.float32)
        dqn_ma = moving_average(rewards_dqn_arr, window)

        all_rewards_dqn.append(rewards_dqn_arr)
        all_rewards_rmmert.append(rewards_rmmert_arr)

        if rmmert_target["state"] is not None:
            torch.save(
                rmmert_target["state"],
                f"rmmert_reward{int(target_reward)}_run{run_idx}_ep{rmmert_target['episode']}.pt",
            )
        if dqn_at_rmmert_target["state"] is not None:
            torch.save(
                dqn_at_rmmert_target["state"],
                f"dqn_at_rmmert_reward{int(target_reward)}_run{run_idx}_ep{dqn_at_rmmert_target['episode']}.pt",
            )
        if best_eval_rmmert["state"] is not None:
            torch.save(
                best_eval_rmmert["state"],
                f"rmmert_best_eval_run{run_idx}_ep{best_eval_rmmert['episode']}.pt",
            )
        if best_eval_dqn["state"] is not None:
            torch.save(
                best_eval_dqn["state"],
                f"dqn_best_eval_run{run_idx}_ep{best_eval_dqn['episode']}.pt",
            )

        np.save(f"rewards_dqn_run{run_idx}.npy", rewards_dqn_arr)
        np.save(f"rewards_rmmert_run{run_idx}.npy", rewards_rmmert_arr)
        np.save(f"rewards_dqn_run{run_idx}_ma10.npy", moving_average(rewards_dqn_arr, 10))
        np.save(f"rewards_rmmert_run{run_idx}_ma10.npy", moving_average(rewards_rmmert_arr, 10))

        env_dqn.close()
        env_rmmert.close()
        eval_env_dqn.close()
        eval_env_rmmert.close()

    # Aggregate across runs
    rewards_dqn_stack = np.stack(all_rewards_dqn, axis=0)
    rewards_rmmert_stack = np.stack(all_rewards_rmmert, axis=0)

    mean_dqn = rewards_dqn_stack.mean(axis=0)
    std_dqn = rewards_dqn_stack.std(axis=0)
    mean_rmmert = rewards_rmmert_stack.mean(axis=0)
    std_rmmert = rewards_rmmert_stack.std(axis=0)

    np.save("rewards_dqn_mean.npy", mean_dqn)
    np.save("rewards_dqn_std.npy", std_dqn)
    np.save("rewards_rmmert_mean.npy", mean_rmmert)
    np.save("rewards_rmmert_std.npy", std_rmmert)
    np.save("rewards_dqn_mean_ma10.npy", moving_average(mean_dqn, 10))
    np.save("rewards_rmmert_mean_ma10.npy", moving_average(mean_rmmert, 10))

    plot_rewards(
        rewards_series=[mean_dqn, mean_rmmert],
        labels=["DQN (mean of runs)", "DQN + ReMERT (mean of runs)"],
        save_path="rewards.png",
        window=10,
        std_series=[std_dqn, std_rmmert],
    )
    print("Plot salvato in rewards.png; salvataggi .npy per raw/mean/std completati.")


if __name__ == "__main__":
    main()
