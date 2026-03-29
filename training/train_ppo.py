"""
IntelliCredit PPO Training Baseline
====================================
Train a PPO agent on IntelliCredit-CreditAppraisal-v1 (Task 1 - Easy)
and generate a learning curve to prove learnability.

Usage:
    pip install -r training/requirements.txt
    python training/train_ppo.py

Output:
    training/learning_curve.png
    training/ppo_intellicredit/  (saved model)

GAP 8: Proves the environment is learnable via RL.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from server.intellicredit_env import IntelliCreditEnvironment, TASK_CONFIGS
from models import IntelliCreditAction


class IntelliCreditGymWrapper(gym.Env):
    """Wraps IntelliCreditEnvironment into a gymnasium.Env for SB3 training."""

    metadata = {"render_modes": []}

    def __init__(self, task_id: str = "task1"):
        super().__init__()
        self._env = IntelliCreditEnvironment(task_id=task_id)
        self._task_id = task_id

        # 45-dim observation: 25 app + 10 portfolio + 5 macro + 5 alerts
        # Bounds: [-1, 1] to accommodate missing data sentinel (-1.0)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(45,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def _obs_to_array(self, obs) -> np.ndarray:
        """Convert IntelliCreditObservation to flat numpy array."""
        vec = (
            obs.application_features    # 25
            + obs.portfolio_state        # 10
            + obs.macro_state            # 5
            + obs.alert_state            # 5
        )
        return np.array(vec, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset(seed=seed)
        return self._obs_to_array(obs), {}

    def step(self, action):
        action_obj = IntelliCreditAction(decision=int(action))
        obs = self._env.step(action_obj)

        reward = obs.reward
        terminated = obs.done
        truncated = False
        info = {
            "episode_score": obs.episode_score,
            "reward_components": obs.reward_components,
        }

        return self._obs_to_array(obs), reward, terminated, truncated, info


class RewardLogger(BaseCallback):
    """Log episode rewards for learning curve generation."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0.0
        self._current_length = 0

    def _on_step(self) -> bool:
        self._current_reward += self.locals.get("rewards", [0])[0]
        self._current_length += 1

        # Check if episode ended
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0

            if len(self.episode_rewards) % 100 == 0:
                recent = self.episode_rewards[-50:]
                avg = sum(recent) / len(recent)
                print(f"  Episodes: {len(self.episode_rewards)}, Avg Reward (last 50): {avg:.2f}")

        return True


def plot_learning_curve(rewards, output_path="training/learning_curve.png"):
    """Generate and save the learning curve graph."""
    if len(rewards) < 10:
        print("Not enough episodes to plot a learning curve.")
        return

    # Compute rolling average
    window = min(50, len(rewards) // 4)
    if window < 2:
        window = 2
    rolling_avg = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        rolling_avg.append(sum(rewards[start:i+1]) / (i - start + 1))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Raw rewards (transparent)
    ax.plot(rewards, alpha=0.15, color="#4A90D9", linewidth=0.5, label="Episode Reward")

    # Rolling average (solid)
    ax.plot(rolling_avg, color="#E63946", linewidth=2.0, label=f"Rolling Avg ({window} ep)")

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Total Episode Reward", fontsize=13)
    ax.set_title(
        "IntelliCredit-CreditAppraisal-v1 — PPO Learning Curve (Task 1)",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate start and end
    if len(rewards) > 50:
        start_avg = sum(rewards[:20]) / 20
        end_avg = sum(rewards[-20:]) / 20
        ax.annotate(
            f"Start: {start_avg:.1f}",
            xy=(10, start_avg), fontsize=10, color="#666",
            arrowprops=dict(arrowstyle="->", color="#666"),
            xytext=(len(rewards)*0.15, start_avg + 2),
        )
        ax.annotate(
            f"End: {end_avg:.1f}",
            xy=(len(rewards)-10, end_avg), fontsize=10, color="#666",
            arrowprops=dict(arrowstyle="->", color="#666"),
            xytext=(len(rewards)*0.7, end_avg + 2),
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Learning curve saved to {output_path}")


def main():
    print("=" * 60)
    print("  IntelliCredit PPO Training")
    print("  Task: task1 (Easy — Clean profiles)")
    print("=" * 60)

    # Create environment
    env = IntelliCreditGymWrapper(task_id="task1")

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device="cuda",  # Explicitly force GPU
    )

    # Train with callback
    callback = RewardLogger()
    total_timesteps = 500_000  # Full run for learnability test (500k steps = ~100k episodes for task1)

    print(f"\n  Training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    model_path = "training/ppo_intellicredit"
    model.save(model_path)
    print(f"  Model saved to {model_path}")

    # Plot learning curve
    plot_learning_curve(callback.episode_rewards)

    # Print final stats
    if callback.episode_rewards:
        last_50 = callback.episode_rewards[-50:]
        print(f"\n  Final average reward (last 50 episodes): {sum(last_50)/len(last_50):.2f}")
        print(f"  Total episodes: {len(callback.episode_rewards)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
