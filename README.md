# LlamaTrader: OpenEnv Algorithmic Daytrading Simulator

## 🎯 Environment Description & Motivation
LlamaTrader is a high-fidelity algorithmic daytrading environment built strictly to the OpenEnv specification. 

**Motivation:** Most Reinforcement Learning trading environments rely on abstract, randomized numerical arrays or simplified games. LlamaTrader bridges the gap to reality by simulating a highly realistic intraday quantitative trading desk. It challenges AI agents to maximize Net Asset Value (NAV) using actual historical 5-minute stock data (META), augmented with a "Golden Suite" of professional technical indicators. The goal is to evaluate an LLM's ability to interpret complex, multi-dimensional time-series data and execute regime-aware trading strategies (e.g., Mean Reversion vs. Trend Following) without hallucinating or breaking structural JSON constraints.

## 🧩 Space Definitions

**Observation Space:** A strictly typed Pydantic model representing a single 5-minute market state.
* `cash` (float): Available buying power.
* `shares` (int): Current asset holdings.
* `price` (float): Current asset price.
* `ema_20` (float): 20-period Exponential Moving Average (Trend indicator).
* `rsi_14` (float): 14-period Relative Strength Index (Momentum indicator).
* `macd_histogram` (float): MACD Histogram (Momentum/Trend indicator).
* `bb_lower` & `bb_upper` (float): Bollinger Bands (Volatility bounds).
* `vwap` (float): Volume Weighted Average Price (Volume/Trend indicator).

**Action Space:**
A strictly typed Pydantic model enforcing bounded trading mechanics. The environment applies a strict 0.15% transaction fee to penalize infinite looping.
* `action_type` (Literal): Strict string choice of `"buy"`, `"sell"`, or `"hold"`.
* `percentage` (float): The proportion of cash to spend (if buying) or shares to liquidate (if selling), bounded strictly between `0.0` and `1.0`.

## 📊 Task Descriptions & Expected Difficulty

The environment evaluates agents across a curriculum of three distinct market regimes. Each task features a custom programmatic grader that outputs a deterministic `0.0` to `1.0` score based on the agent's final NAV compared to a Buy-and-Hold baseline.

1. **Capital Preservation (Easy)**
   * **Scenario:** A highly volatile, sideways market day.
   * **Objective:** Survive the volatility without suffering severe drawdowns.
   * **Grading:** Scores 1.0 for maintaining or slightly growing capital (+2%); scores 0.0 for losing 2% or more.
2. **Bear Market Navigation (Medium)**
   * **Scenario:** A severe intraday market crash.
   * **Objective:** Recognize the downtrend and protect capital by liquidating to cash.
   * **Grading:** Scores 0.0 if the agent loses as much as the market; scores 1.0 if it perfectly hedges and loses $0.
3. **Alpha Generation (Hard)**
   * **Scenario:** A massive intraday Bull Market rally.
   * **Objective:** Actively trade to beat a standard "Buy and Hold" strategy. 
   * **Grading:** Scores 0.0 if the agent simply holds or underperforms the baseline; scores 1.0 for beating the baseline by 2% through active trading.

## 🏆 Baseline Scores
These baseline scores were achieved using Meta's `llama-3-8b-8192` model via the Groq API, operating with a zero-shot, prompt-engineered policy. 

* **Task 1 (Easy):** `0.522 / 1.000` (Successfully mitigated volatility)
* **Task 2 (Medium):** `0.048 / 1.000` (Struggled to override RSI momentum signals during the crash)
* **Task 3 (Hard):** `0.000 / 1.000` (Failed to beat a pure Buy-and-Hold strategy factoring in transaction fees)

## 🗂️ Project Structure
* `models.py`: Pydantic schemas defining the strictly typed Observation, Action, and Reward spaces.
* `env.py`: The core `DayTraderEnv` engine handling state transitions, portfolio math, and fees.
* `tasks.py`: Contains the three market regimes and their specific grading logic.
* `download_data.py`: Utility script to fetch historical data via `yfinance` and calculate indicators via `pandas-ta`.
* `inference.py`: The baseline runner connecting to Llama 3 and executing the simulation loop.
* `data/`: Directory containing static, pre-processed market CSV files to ensure deterministic evaluation.
* `openenv.yaml`: Metadata configuration for OpenEnv compliance.
* `Dockerfile` & `.dockerignore`: Containerization settings for cloud deployment.

## 🚀 Setup & Usage Instructions

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/CandiedOutlaw763/rl_environment
cd rl_environment
pip install -r requirements.txt

### ⚠️ Note for Judges regarding API Rate Limits
This baseline script executes over 230 individual LLM calls to complete the 3-task curriculum. Because it relies on the free tier of the Groq API, running the script multiple times in rapid succession may trigger a `RateLimitError` (e.g., "token limit for current day exceeded" or "requests per minute exceeded"). 

If you encounter this error during evaluation:
1. Please wait 1-2 minutes for the minute-bound token bucket to reset.
2. Alternatively, you can swap the `GROQ_API_KEY` in the `.env` file (or Hugging Face Space secrets) with your own Groq key.
