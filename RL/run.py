import sys
import os

# --- PATH SETUP ---
# Get current directory (RL/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get tasks directory (../tasks)
tasks_dir = os.path.join(os.path.dirname(current_dir), 'tasks')
# Get parent directory (for searching CSVs or other utils if needed)
parent_dir = os.path.dirname(current_dir)

# Add them to system path
sys.path.append(tasks_dir)
sys.path.append(parent_dir)
# ------------------

import torch
from embeddings import QueryEmbedder
from env import HierarchicalTaskEnv
from policy import HierarchicalPPO

def run_agent():
    print("--- Initializing System ---")
    
    # 1. Load Components
    embedder = QueryEmbedder()
    env = HierarchicalTaskEnv(embedder)
    
    state_dim = 128  # 128 from BERT
    action_dims = [2, 10]
    
    # 2. Load Model
    model = HierarchicalPPO(state_dim, action_dims)
    try:
        model.load_state_dict(torch.load("ppo_agent.pth"))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No saved model found! Running with random weights.")
    
    model.eval()

    print("\n--- AI Agent Ready ---")
    print("Tasks available: 1. Temperature, 2. ET0 (Evapotranspiration)")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == 'exit':
            break
            
        # 3. Process Input
        state, _ = env.reset(user_query)
        
        # 4. Agent Decision
        with torch.no_grad():
            task, days, _, _ = model.act(state, deterministic=True)
            
        # Map actions to readable names for debugging
        task_name = "Temperature" if task == 0 else "ET0"
        days_value = days + 1
        
        print(f"\n[Agent Logic] Embedding generated.")
        print(f"[Agent Logic] Selected Task: {task_name}")
        print(f"[Agent Logic] Selected Param: {days_value} days")
        
        # 5. Execute in Environment
        env.step([task, days])
        print("-" * 50)

if __name__ == "__main__":
    run_agent()