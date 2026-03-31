import os
import json
import pandas as pd
from openai import OpenAI
from pydantic import ValidationError

# Import our custom logic
from models import Action
from tasks import CapitalPreservationTask, AlphaGenerationTask, BearMarketTask
from env import DayTraderEnv

# 1. Setup OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key)

# 2. Load the Static Datasets
print("Loading Market Data from CSVs...")
try:
    bull_df = pd.read_csv("data/bull_market.csv", index_col=0)
    sideways_df = pd.read_csv("data/sideways_market.csv", index_col=0)
    bear_df = pd.read_csv("data/bear_market.csv", index_col=0)
except FileNotFoundError:
    raise FileNotFoundError("Could not find the CSV files. Please run download_data.py first.")

# Initialize the three required tasks
tasks = [
    CapitalPreservationTask(market_data=sideways_df),
    AlphaGenerationTask(market_data=bull_df),
    BearMarketTask(market_data=bear_df)
]

def get_agent_action(observation_json: str) -> Action:
    """Calls the LLM with a 'Cheat Sheet' prompt and forces a valid Action."""
    
    # The 'Secret Sauce': We tell the LLM exactly how to interpret the Golden Suite
    system_prompt = (
        "You are an elite algorithmic daytrader. Your goal is to maximize Net Asset Value (NAV).\n"
        "You will receive 5-minute interval market data with technical indicators.\n"
        "--- INDICATOR CHEAT SHEET ---\n"
        "* RSI_14: > 70 indicates Overbought (consider selling). < 30 indicates Oversold (consider buying).\n"
        "* VWAP: If Current Price > VWAP, the intraday trend is Bullish. If Price < VWAP, it is Bearish.\n"
        "* Bollinger Bands: Price touching 'bb_lower' implies strong support/reversal upwards. Price touching 'bb_upper' implies resistance.\n"
        "* MACD Histogram: Positive values indicate growing bullish momentum. Negative is bearish.\n"
        "* EMA_20: Acts as a dynamic baseline for the short-term trend.\n"
        "-----------------------------\n"
        "You must respond in strict JSON matching this schema:\n"
        "{'action_type': 'buy'|'sell'|'hold', 'percentage': float between 0.0 and 1.0}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Current Market State: {observation_json}"}
            ],
            temperature=0.1 # Low temperature keeps the AI focused on the math, not hallucinating
        )
        
        raw_response = response.choices[0].message.content
        parsed_json = json.loads(raw_response)
        
        # Pydantic validates the response against models.py immediately
        return Action(**parsed_json)
        
    except (ValidationError, json.JSONDecodeError, Exception) as e:
        # Failsafe: If the LLM breaks, default to holding so the simulation doesn't crash
        return Action(action_type="hold", percentage=0.0)

# 3. Run the Evaluation Loop
print("\nStarting OpenEnv Baseline Inference...\n")

for task in tasks:
    print(f"==================================================")
    print(f" R U N N I N G   T A S K :   {task.name} ({task.difficulty})")
    print(f"==================================================")
    
    # Initialize the environment with the specific task's DataFrame
    env = DayTraderEnv(market_data=task.market_data)
    obs = env.reset()
    
    done = False
    step_count = 0
    
    while not done:
        # 1. Convert the current Pydantic state to JSON
        obs_json = obs.model_dump_json()
        
        # 2. Ask the LLM for its next move
        action = get_agent_action(obs_json)
        
        # 3. Step the environment forward
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # Optional: Print every 15 steps just to watch it work
        if step_count % 15 == 0:
            print(f"Step {step_count}: Price: ${obs.price:.2f} | Action: {action.action_type.upper()} {action.percentage*100:.0f}% | NAV: ${(env.cash + env.shares * obs.price):.2f}")
            
    # Episode complete. Grade the final NAV.
    final_nav = env.cash + (env.shares * obs.price)
    score = task.grade(final_nav)
    
    print(f"\n--- TASK COMPLETE ---")
    print(f"Total Steps Executed: {step_count}")
    print(f"Starting Cash: ${env.initial_cash:.2f}")
    print(f"Final NAV:     ${final_nav:.2f}")
    print(f"Total Return:  {((final_nav - env.initial_cash) / env.initial_cash) * 100:.2f}%")
    print(f"FINAL SCORE:   {score:.3f} / 1.000\n")

print("All baselines completed successfully. Ready for submission.")