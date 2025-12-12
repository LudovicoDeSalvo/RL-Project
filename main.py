import gym
import numpy as np

from dqn_agent import DQNAgent, ReplayBuffer
from utils import plot_rewards, set_seed


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

        if (episode + 1) % 25 == 0:
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"{'ReMERT ' if use_rmmert else ''}Reward: {total_reward:.1f}"
            )
    return rewards


def main():
    set_seed(42)
    env_name = "CartPole-v1"
    env_seed = 42
    episodes = 250
    batch_size = 64
    warmup_steps = 1_000
    buffer_capacity = 50_000

    env_dqn = create_env(env_name, seed=env_seed)
    env_rmmert = create_env(env_name, seed=env_seed)

    state_dim = env_dqn.observation_space.shape[0]
    action_dim = env_dqn.action_space.n

    dqn_agent = DQNAgent(state_dim, action_dim)
    dqn_buffer = ReplayBuffer(buffer_capacity)
    rmmert_agent = DQNAgent(state_dim, action_dim)
    rmmert_buffer = ReplayBuffer(buffer_capacity)

    if warmup_steps > 0:
        print(f"Warmup buffer DQN with {warmup_steps} random steps...")
        warmup_buffer(env_dqn, dqn_buffer, warmup_steps, use_rmmert=False)
        print(f"Warmup buffer ReMERT with {warmup_steps} random steps...")
        warmup_buffer(env_rmmert, rmmert_buffer, warmup_steps, use_rmmert=True)

    print("Training standard DQN...")
    rewards_dqn = run_agent(
        env_dqn, dqn_agent, dqn_buffer, episodes, batch_size, warmup_steps, use_rmmert=False
    )

    print("\nTraining DQN + ReMERT (prioritized on distance to end)...")
    rewards_rmmert = run_agent(
        env_rmmert, rmmert_agent, rmmert_buffer, episodes, batch_size, warmup_steps, use_rmmert=True
    )

    plot_rewards(
        rewards_series=[rewards_dqn, rewards_rmmert],
        labels=["DQN", "DQN + ReMERT"],
        save_path="rewards.png",
        window=10,
    )
    np.save("rewards_dqn.npy", np.array(rewards_dqn, dtype=np.float32))
    np.save("rewards_rmmert.npy", np.array(rewards_rmmert, dtype=np.float32))
    print("Plot salvato in rewards.png")

    env_dqn.close()
    env_rmmert.close()


if __name__ == "__main__":
    main()
