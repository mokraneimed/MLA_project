import torch
import torch.nn as nn
from torch.distributions import Categorical

class HierarchicalPPO(nn.Module):
    def __init__(self, state_dim, action_dims):
        super(HierarchicalPPO, self).__init__()
        
        # Shared Feature Extractor
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # --- ACTOR HEADS ---
        self.actor_task = nn.Linear(64, action_dims[0])
        
        self.actor_days = nn.Linear(64, action_dims[1])
        
        # --- CRITIC HEAD ---
        self.critic = nn.Linear(64, 1)

    def act(self, state, deterministic=False):
        state = torch.from_numpy(state).float().unsqueeze(0) # Add batch dim
        features = self.backbone(state)
        
        # Get Logits
        task_logits = self.actor_task(features)
        days_logits = self.actor_days(features)

        if deterministic:
            # [CHANGE] Greedy selection for Evaluation
            # We take the index with the highest probability
            task_action = torch.argmax(task_logits, dim=-1)
            days_action = torch.argmax(days_logits, dim=-1)
            
            # Return zeros for log_prob/value as they aren't used in eval
            return task_action.item(), days_action.item(), 0, None
        else:
            # [CHANGE] Stochastic selection for Training
            task_dist = Categorical(logits=task_logits)
            days_dist = Categorical(logits=days_logits)
            
            task_action = task_dist.sample()
            days_action = days_dist.sample()
            
            # Calculate log probs for PPO
            action_logprob = task_dist.log_prob(task_action) + days_dist.log_prob(days_action)
            
            return task_action.item(), days_action.item(), action_logprob, self.critic(features)

    def evaluate(self, state, task_action, days_action):
        # Used during training loop
        features = self.backbone(state)
        
        # Task
        task_logits = self.actor_task(features)
        task_dist = Categorical(logits=task_logits)
        
        # Days
        days_logits = self.actor_days(features)
        days_dist = Categorical(logits=days_logits)
        
        # Log Probs & Entropy
        task_logprobs = task_dist.log_prob(task_action)
        days_logprobs = days_dist.log_prob(days_action)
        
        action_logprobs = task_logprobs + days_logprobs
        dist_entropy = task_dist.entropy() + days_dist.entropy()
        state_values = self.critic(features)
        
        return action_logprobs, state_values, dist_entropy