# ReMERT: Enhancing Learning Efficiency in Deep Q-Learning

## Project Overview
This project implements and evaluates **ReMERT** (Replay Memory with End-Related Transitions), a modification to the standard Experience Replay mechanism in Deep Q-Networks (DQN). The study compares standard DQN against DQN + ReMERT on the `CartPole-v1` environment.

The core hypothesis is that prioritizing transitions closer to the end of an episode (terminal state) provides valuable credit assignment information, leading to faster convergence and improved learning stability in episodic tasks.


## Algorithm: ReMERT
ReMERT modifies the sampling probability of transitions in the replay buffer based on their temporal proximity to the end of the episode.

### Weighting Strategy
Standard Experience Replay samples uniformly. ReMERT assigns importance weights ($w$) to transitions using the following inverse-distance formula:

$$w \propto \frac{1}{\text{distance\\_to\\_end} + 1}$$

Where `distance_to_end` is the number of steps remaining in the episode from the current state. This ensures that states leading up to success or failure are sampled more frequently, accelerating credit assignment.


## Project Structure
* **`dqn_agent.py`**: Implementation of `QNetwork`, `DQNAgent`, and the modified `ReplayBuffer`. The buffer handles the `distance_to_end` calculation and weighted sampling.
* **`main.py`**: Main entry point. Runs comparative training (DQN vs. ReMERT) over multiple random seeds, logs performance, and saves models/data.
* **`plot_graphs.py`**: Loads generated `.npy` data to create detailed performance plots (rewards, convergence time, distributions).
* **`utils.py`**: Helper functions for moving averages, seeding, and plotting.
* **`watch_policy.py`**: Script to render and watch a trained agent in real-time.
* **`visualize.py`**: Script to record video of a trained agent.

## Usage

### 1. Training
Run the main training loop. This will train both Standard DQN and ReMERT agents across 3 different seeds (by default) for 1000 episodes each.

```bash
python main.py

```

**Output:**

* `.pt` files: Saved model checkpoints (e.g., best evaluation, target reward reached).
* `.npy` files: Raw reward data for analysis.
* `rewards.png`: Initial comparison plot.

### 2. Analysis & Plotting

Generate detailed comparison graphs after training is complete.

```bash
python plot_graphs.py

```

**Generated Plots:**

* `rewards_split_*.png`: Separate learning curves for readability.
* `convergence_time.png`: Bar chart comparing episodes to reach solution threshold.
* `early_training_rewards.png`: Zoomed view of the first 200-300 episodes.
* `reward_distribution.png`: Histogram of rewards in late training.

### 3. Visualization

To watch a specific trained model (replace filename with your specific checkpoint):

```bash
python watch_policy.py --checkpoint rmmert_best_eval_run1_epXXX.pt --env CartPole-v1

```

To record a video:

```bash
python visualize.py --checkpoint rmmert_best_eval_run1_epXXX.pt --out my_video_prefix

```

## Key Results

Based on the experimental data included in the project:

* **Sample Efficiency:** ReMERT demonstrates a clear advantage in the first 300 episodes, learning crucial milestones faster than standard DQN.


* **Stability:** ReMERT exhibits less variance in performance across different runs and maintains higher consistency even when overfitting risks arise in later episodes.


* **Convergence:** ReMERT generally achieves the target reward threshold (475.0) in fewer episodes compared to the baseline.



## Hyperparameters

* **Environment:** CartPole-v1
* **Hidden Layers:** [128, 128]
* **Learning Rate:** 1e-3
* **Batch Size:** 64
* **Buffer Capacity:** 50,000
* **Gamma:** 0.99
* **Warmup Steps:** 1,000