import random
from typing import Iterable, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    # Simple incremental average; fast enough for hundreds of episodes. For very long runs, np.cumsum is faster.
    if window <= 1:
        return np.array(values, dtype=np.float32)
    values_array = np.array(values, dtype=np.float32)
    result: List[float] = []
    running = 0.0
    for idx, value in enumerate(values_array):
        running += value
        if idx + 1 > window:
            running -= values_array[idx - window]
            result.append(running / window)
        else:
            result.append(running / (idx + 1))
    return np.array(result, dtype=np.float32)


def plot_rewards(
    rewards_series: List[Sequence[float]],
    labels: Sequence[str],
    save_path: str = "rewards.png",
    window: int = 10,
    title: str = "CartPole - DQN vs ReMERT",
) -> None:
    plt.figure(figsize=(10, 5))
    for rewards, label in zip(rewards_series, labels):
        smoothed = moving_average(rewards, window)
        plt.plot(rewards, alpha=0.3, label=f"{label} (raw)")
        plt.plot(smoothed, linewidth=2, label=f"{label} (avg {window})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
