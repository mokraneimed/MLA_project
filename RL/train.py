import torch
import torch.nn as nn
import torch.optim as optim
from embeddings import QueryEmbedder
from env import HierarchicalTaskEnv
from policy import HierarchicalPPO

# Hyperparameters
LR = 0.002
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
EPISODES = 50 

def train():
    # 1. Setup
    embedder = QueryEmbedder()
    env = HierarchicalTaskEnv(embedder)
    
    state_dim = 128
    action_dims = [2, 10] # [Tasks, Days]
    
    policy = HierarchicalPPO(state_dim, action_dims)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    policy_old = HierarchicalPPO(state_dim, action_dims)
    policy_old.load_state_dict(policy.state_dict())
    
    mse_loss = nn.MSELoss()
    
    # Dummy Training Data
    training_queries = [
        "Predict temperature for 5 days", "Tell me EUV for next week", 
        "What is the temp?", "EUV forecast needed", 
        "Temperature prediction required", "EUV data for 10 days"
    ]

    print("Starting Training...")

    for i in range(EPISODES):
        # Select random query to simulate variety
        query = training_queries[i % len(training_queries)]
        
        # --- 1. Run Old Policy to collect data ---
        state, _ = env.reset(query)
        task, days, log_prob, _ = policy_old.act(state)
        
        _, reward, done, _, _ = env.step([task, days])
        
        # Convert to tensors
        old_states = torch.FloatTensor(state).unsqueeze(0)
        old_task_actions = torch.tensor([task])
        old_days_actions = torch.tensor([days])
        old_logprobs = log_prob.detach()
        rewards = torch.tensor([reward], dtype=torch.float32).view(-1, 1)

        # --- 2. Update Policy ---
        # Since it's 1-step episode, Advantage = Reward - Value (simplified)
        
        for _ in range(K_EPOCHS):
            # Evaluate current policy
            logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_task_actions, old_days_actions)
            
            # Calculate Advantage
            # Ideally use GAE, but for 1-step: Advantage = Reward - Baseline
            advantages = rewards - state_values.detach()
            
            # Ratio for PPO
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            
            # Total Loss: -Actor + 0.5*Critic - 0.01*Entropy
            loss = -torch.min(surr1, surr2) + 0.5 * mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            
        # Update old policy
        policy_old.load_state_dict(policy.state_dict())
        
        if i % 10 == 0:
            print(f"Episode {i}/{EPISODES} completed. Last Reward: {reward:.4f}")

    # Save Model
    torch.save(policy.state_dict(), "ppo_agent.pth")
    print("Training finished. Model saved to ppo_agent.pth")

if __name__ == "__main__":
    train()