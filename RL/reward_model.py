import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

# 1. Load the .env file immediately
load_dotenv()

# 2. Get the key
API_KEY = os.getenv("GROQ_API_KEY")

class LLMRewardModel:
    def __init__(self):
        if not API_KEY:
            print("ERROR: GROQ_API_KEY not found. Check your .env file.")
            self.client = None
        else:
            self.client = Groq(api_key=API_KEY)
            
        self.model_name = "llama-3.3-70b-versatile"

        ## cache
        self.cache_file = "reward_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=4)        

    def get_reward(self, user_query, agent_task, agent_days):
        true_task = None
        true_days = None

        # --- STEP 1: GET GROUND TRUTH (Cache or API) ---
        if user_query in self.cache:
            ground_truth = self.cache[user_query]
            print("Using cached reward data.")
            true_task = ground_truth['true_task_id']
            true_days = ground_truth['true_days']

        else:    
            if not self.client:
                return 0.0, {"error": "No API Key"}

            prompt = f"""
            You are an AI Judge for a Reinforcement Learning agent.
            
            CONTEXT:
            - Task ID 0: Temperature Prediction
            - Task ID 1: ET0 (Evapotranspiration) Prediction
            
            USER QUERY: "{user_query}"
            
            YOUR JOB:
            Extract the INTENT from the query:
            1. What is the correct Task ID? (0 or 1)
            2. What is the requested number of days? (Integer only. If not specified, default to 1).
            
            OUTPUT FORMAT:
            Return ONLY a JSON object. No explanations.
            Example: {{"true_task_id": 0, "true_days": 5}}
            """

            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs strictly JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=0.0,
                )
                
                response_text = chat_completion.choices[0].message.content
                
                # Robust JSON parsing
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    ground_truth = json.loads(json_str)
                    
                    true_task = int(ground_truth.get('true_task_id', 0))
                    true_days = int(ground_truth.get('true_days', 1))

                    ## update cache
                    self.cache[user_query] = {
                        'true_task_id': true_task,
                        'true_days': true_days
                    }
                    self._save_cache()
                else:
                    return 0.0, {"error": "JSON Parse Error"}

            except Exception as e:
                print(f"Groq API Error: {e}")
                return 0.0, {"error": str(e)}

        # --- STEP 2: CALCULATE REWARD (Common Logic) ---
        # This now runs for BOTH Cache hits and API calls
        
        # R1: Task Accuracy
        if agent_task == true_task:
            r1 = 30.0
        else:
            r1 = -30.0
        
        # R2: Input Precision (Negative Error)
        r2 = -float(abs(true_days - agent_days))
        
        total_reward = r1 + r2
        
        return total_reward, {
            "r1": r1,
            "r2": r2,
            "true_task": true_task,
            "true_days": true_days,
            "agent_task": agent_task,
            "agent_days": agent_days
        }

if __name__ == "__main__":
    rw = LLMRewardModel()
    
    # 1. Test to see if key loads
    if rw.client:
        print("✅ API Key loaded successfully!")
    else:
        print("❌ Failed to load API Key. Exiting test.")
        exit()

    # 2. Run a real Prompt Test
    print("\n--- Running Functional Test ---")
    
    test_query = "Please predict temperature for 8 days"
    simulated_task = 0 
    simulated_days = 3 
    
    print(f"Query: '{test_query}'")
    print(f"Simulated Agent Action: Task={simulated_task}, Days={simulated_days}")
    
    score, info = rw.get_reward(test_query, simulated_task, simulated_days)
    
    print(f"\nResult:")
    print(f"  > Total Score: {score}") 
    print(f"  > Detailed Info: {info}")