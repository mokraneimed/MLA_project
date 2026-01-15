import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from reward_model import LLMRewardModel
from tasks.forecaster import TaskForecaster

class HierarchicalTaskEnv(gym.Env):
    def __init__(self, embedder):
        super(HierarchicalTaskEnv, self).__init__()
        self.embedder = embedder
        self.reward_model = LLMRewardModel()
        self.forecaster = TaskForecaster()

        # State: BERT embedding size (128)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)
        
        # Action Space: MultiDiscrete
        # Index 0: Task Selection (0 = Temperature, 1 = EUV)
        # Index 1: Input Parameter (Days) - Let's allow 1 to 10 days (mapped from 0-9)
        self.action_space = spaces.MultiDiscrete([2, 10])
        
        self.current_query = None

    def reset(self, query=None, seed=None):
        super().reset(seed=seed)
        if query is None:
            # Fallback for random training steps
            query = "Default system check"
        
        self.current_query = query
        state = self.embedder.get_embedding(query)
        return state, {}

    def step(self, action):
        """
        action is an array: [task_id, days_index]
        """
        task_id = action[0]
        days_input = action[1] + 1  # Convert 0-9 index to 1-10 days
        
        predictions = self.forecaster.predict(task_id, days_input)
        task_name = "Temperature (°C)" if task_id == 0 else "ET0 (mm)"
        print(f"\n[System] TASK: {task_name}")
        print(f"[System] FORECAST ({days_input} days):")

        pred_str = ", ".join([f"{p:.2f}" for p in predictions])
        print(f"   > {pred_str}")
            
        # Reward
        reward, info = self.reward_model.get_reward(
            self.current_query, task_id, days_input
        )
        if "r1" in info:
            print(f"   [Reward] Total: {reward:.1f} | R1 (Task): {info['r1']} | R2 (Days Diff): {info['r2']}")
        # In a one-shot task (query -> response), the episode ends immediately
        terminated = True 
        truncated = False
        
        return self.observation_space.sample(), reward, terminated, truncated, info