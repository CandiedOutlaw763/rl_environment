import os
import sys

# Ensure the root directory is in the Python path so it can find env.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.server import serve
from env import DayTraderEnv

def main():
    # Initialize your environment and bind it to the Hugging Face port
    env = DayTraderEnv()
    serve(env, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()