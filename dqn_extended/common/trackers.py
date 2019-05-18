import sys
import time
import numpy as np
import wandb


class RewardTracker:
    def __init__(self, stop_reward):
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        pass

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])

        logs = {}
        if epsilon is not None:
            logs["epsilon"] = epsilon

        logs["speed"] = speed
        logs["reward_100"] = mean_reward
        logs["reward"] = reward

        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True, logs

        return False, logs


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params["epsilon_start"]
        self.epsilon_final = params["epsilon_final"]
        self.epsilon_frames = params["epsilon_frames"]
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = max(
            self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames
        )
