import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class HierarchicalTaskEnv(gym.Env):
    def __init__(self, embedder):
        super(HierarchicalTaskEnv, self).__init__()
        self.embedder = embedder
        
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
        
        # Execute the specific task logic
        if task_id == 0:
            self._run_temperature_task(days_input)
        else:
            self._run_euv_task(days_input)
            
        # Reward Logic: Random for now, as requested
        reward = np.random.randn() 
        
        # In a one-shot task (query -> response), the episode ends immediately
        terminated = True 
        truncated = False
        
        return self.observation_space.sample(), reward, terminated, truncated, {}

    def _run_temperature_task(self, days):
        # Static output mock
        print(f"\n[System] TASK: Temperature Prediction")
        print(f"[System] INPUT: {days} days")
        print(f"[Result] The predicted temperature for the next {days} days are: 24°C, 25°C, 23°C ... (static mock)")

    def _run_euv_task(self, days):
        # Static output mock
        print(f"\n[System] TASK: EUV (Extreme Ultraviolet) Prediction")
        print(f"[System] INPUT: {days} days")
        print(f"[Result] The predicted EUV flux for the next {days} days are: 1.2e-3, 1.4e-3 ... (static mock)")