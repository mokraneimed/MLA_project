import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
tasks_dir = os.path.join(os.path.dirname(current_dir), 'tasks')
parent_dir = os.path.dirname(current_dir)
sys.path.append(tasks_dir)
sys.path.append(parent_dir)
# ------------------

import torch
import torch.nn as nn
import torch.optim as optim
from embeddings import QueryEmbedder
from env import HierarchicalTaskEnv
from policy import HierarchicalPPO

# --- OPTIMIZED HYPERPARAMETERS ---
LR = 0.0001              # Strong learning rate for fast convergence
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 10           # Drill the batch hard
EPISODES = 100_000        # 20k is enough for this dataset
UPDATE_TIMESTEP = 2048   # Frequent updates (40 updates total)
MINI_BATCH_SIZE = 128    # Standard mini-batch

# Entropy Decay (Start high to explore, end low to perfect)
ENTROPY_START = 0.01
ENTROPY_END = 0.001

EVAL_EPISODES = 50

def load_training_data():
    dataset_path = os.path.join(current_dir, "training_dataset.txt")
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
        return queries
    return ["Predict temperature for 5 days"]

def evaluate_policy(env, policy, queries):
    total_reward = 0
    # Evaluate on a random subset
    eval_queries = np.random.choice(queries, min(EVAL_EPISODES, len(queries)), replace=True)
    
    for q in eval_queries:
        state, _ = env.reset(q)
        with torch.no_grad():
            task, days, _, _ = policy.act(state, deterministic=True)
        _, reward, _, _, _ = env.step([task, days])
        total_reward += reward
    return total_reward / len(eval_queries)

def plot_training_graphs(loss_history, reward_history_rollout, reward_history_eval):
    epochs = range(len(loss_history['total']))
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f'PPO Training Results', fontsize=16)

    axs[0, 0].plot(epochs, loss_history['total'], 'b', label='Total')
    axs[0, 0].set_title('Total Loss')
    
    axs[0, 1].plot(epochs, loss_history['policy'], 'r', label='Policy')
    axs[0, 1].set_title('Policy Loss')

    axs[1, 0].plot(epochs, loss_history['value'], 'g', label='Value')
    axs[1, 0].set_title('Value Loss')

    axs[1, 1].plot(epochs, loss_history['entropy'], 'm', label='Entropy')
    axs[1, 1].set_title('Entropy')
    
    axs[2, 0].plot(epochs, reward_history_rollout, 'gray', alpha=0.6, label='Rollout')
    axs[2, 0].plot(epochs, reward_history_eval, 'k', linewidth=2, label='Eval')
    axs[2, 0].set_title('Mean Reward')
    axs[2, 0].legend()

    plt.tight_layout()
    plt.savefig("training_results_final.png")
    print("Graphs saved.")

def train():
    embedder = QueryEmbedder()
    env = HierarchicalTaskEnv(embedder)
    
    # 128 input (BERT), 2 tasks, 10 days
    policy = HierarchicalPPO(128, [2, 10])
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    policy_old = HierarchicalPPO(128, [2, 10])
    policy_old.load_state_dict(policy.state_dict())
    
    mse_loss = nn.MSELoss()
    training_queries = load_training_data()

    memory_states = []
    memory_task_actions = []
    memory_days_actions = []
    memory_logprobs = []
    memory_rewards = []

    metrics_loss = {'total': [], 'policy': [], 'value': [], 'entropy': []}
    metrics_reward_rollout = []
    metrics_reward_eval = []

    print(f"Starting Training ({EPISODES} episodes)...")

    for i in range(1, EPISODES + 1):
        query = training_queries[i % len(training_queries)]
        state, _ = env.reset(query)
        
        # Collect Data
        with torch.no_grad():
            task, days, log_prob, _ = policy_old.act(state, deterministic=False)
        
        _, reward, _, _, _ = env.step([task, days])
        
        memory_states.append(state)
        memory_task_actions.append(task)
        memory_days_actions.append(days)
        memory_logprobs.append(log_prob.item())
        memory_rewards.append(reward)

        if i % UPDATE_TIMESTEP == 0:
            print(f"\n[Update] Episode {i}. Updating Policy...")

            # Decay Entropy
            progress = i / EPISODES
            current_ent_coef = ENTROPY_START - (ENTROPY_START - ENTROPY_END) * progress
            current_ent_coef = max(ENTROPY_END, current_ent_coef)

            # Prepare Tensors
            old_states = torch.FloatTensor(np.array(memory_states)).detach()
            old_task_actions = torch.tensor(memory_task_actions).detach()
            old_days_actions = torch.tensor(memory_days_actions).detach()
            old_logprobs = torch.tensor(memory_logprobs).detach()
            
            # Normalize Rewards (Critical for stability)
            raw_rewards = torch.tensor(memory_rewards, dtype=torch.float32).view(-1, 1).detach()
            rewards = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-7)

            metrics_reward_rollout.append(raw_rewards.mean().item())

            # Calculate Advantages Once
            with torch.no_grad():
                _, old_state_values, _ = policy.evaluate(old_states, old_task_actions, old_days_actions)
            
            advantages = rewards - old_state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            # Mini-Batch Update
            dataset_size = len(memory_states)
            indices = np.arange(dataset_size)
            
            last_stats = {}

            for _ in range(K_EPOCHS):
                np.random.shuffle(indices)
                
                for start in range(0, dataset_size, MINI_BATCH_SIZE):
                    end = start + MINI_BATCH_SIZE
                    idx = indices[start:end]

                    logprobs, state_values, dist_entropy = policy.evaluate(
                        old_states[idx], old_task_actions[idx], old_days_actions[idx]
                    )
                    
                    ratios = torch.exp(logprobs - old_logprobs[idx])
                    
                    surr1 = ratios * advantages[idx]
                    surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages[idx]
                    
                    loss_policy = -torch.min(surr1, surr2).mean()
                    loss_value = 0.5 * mse_loss(state_values, rewards[idx])
                    loss_entropy = -current_ent_coef * dist_entropy.mean()

                    loss = loss_policy + loss_value + loss_entropy
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

                    last_stats = {
                        'total': loss.item(), 'policy': loss_policy.item(),
                        'value': loss_value.item(), 'entropy': dist_entropy.mean().item()
                    }

            # Log Metrics
            metrics_loss['total'].append(last_stats['total'])
            metrics_loss['policy'].append(last_stats['policy'])
            metrics_loss['value'].append(last_stats['value'])
            metrics_loss['entropy'].append(last_stats['entropy'])

            # Reset buffers
            policy_old.load_state_dict(policy.state_dict())
            memory_states = []
            memory_task_actions = []
            memory_days_actions = []
            memory_logprobs = []
            memory_rewards = []

            # Evaluate
            eval_score = evaluate_policy(env, policy, training_queries)
            metrics_reward_eval.append(eval_score)
            
            print(f"   > Rollout Avg: {metrics_reward_rollout[-1]:.2f} | Eval Avg: {eval_score:.2f} | EntCoef: {current_ent_coef:.4f}")

    torch.save(policy.state_dict(), "ppo_agent.pth")
    print("Training finished.")
    plot_training_graphs(metrics_loss, metrics_reward_rollout, metrics_reward_eval)

if __name__ == "__main__":
    train()