import os
import time
import re
from groq import Groq
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("Error: GROQ_API_KEY not found in .env file.")
    exit()

client = Groq(api_key=API_KEY)

OUTPUT_FILE = "training_dataset.txt"
TOTAL_QUERIES = 2000
BATCH_SIZE = 100

def generate_batch():
    prompt = f"""
    Generate exactly {BATCH_SIZE} unique natural language queries for a weather forecasting AI.
    
    TOPICS:
    1. Predicting Mean Temperature (use words like: temp, temperature, heat, weather, degrees).
    2. Predicting ET0 (use words like: ET0, evapotranspiration, water loss, crop water need).
    
    DURATION VARIATIONS (Mix these randomly):
    1. Specific Numbers: "for 5 days", "next 9 days", "2 days".
       - You can use Digits (e.g., "5 days") OR Words (e.g., "five days", "ten days").
       - **CONSTRAINT:** The duration MUST be strictly between 1 and 10 days. Do NOT generate queries for more than 10 days.
    2. Natural Language: "for a week" (valid), "tomorrow" (valid).
    3. No Duration (Implicit): "What is the temperature?", "Predict ET0", "Give me the forecast".
    
    CONSTRAINTS:
    - Vary the phrasing (formal, casual, short, long).
    - Do NOT number the lines.
    - Do NOT add introductory text like "Here are the queries".
    - Output ONLY the raw queries, one per line.
    
    EXAMPLES:
    Predict mean temperature for the next 5 days
    What is the expected ET0?
    Give me the temperature forecast for a week
    I need evapotranspiration data for ten days
    Calculate ET0 for 9 days
    How hot will it be?
    Forecast temperature for three days
    """

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a dataset generator. Output only raw text lines."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.9,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return ""

def clean_and_save(text):
    lines = text.strip().split('\n')
    valid_lines = []
    
    for line in lines:
        # Remove "1. ", "- ", etc. if the LLM adds them
        clean_line = re.sub(r'^[\d-]+\.\s*', '', line).strip()
        if clean_line:
            valid_lines.append(clean_line)
            
    # Append to file
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")
            
    return len(valid_lines)

def main():
    # Clear file if exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    count = 0
    batch_num = 1
    
    print(f"--- Starting Data Generation ({TOTAL_QUERIES} queries) ---")
    
    while count < TOTAL_QUERIES:
        print(f"Requesting Batch {batch_num}...", end=" ", flush=True)
        
        raw_text = generate_batch()
        saved = clean_and_save(raw_text)
        
        count += saved
        print(f"Saved {saved} queries. Total: {count}/{TOTAL_QUERIES}")
        
        batch_num += 1
        # Sleep slightly to respect rate limits
        time.sleep(2)

    print(f"\nDone! Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()