import pandas as pd
from typing import Tuple, Dict, Any
from models import Observation, Action, Reward

class DayTraderEnv:
    def __init__(self, market_data: pd.DataFrame, initial_cash: float = 10000.0):
        self.market_data = market_data.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.total_steps = len(self.market_data)
        self.transaction_fee_rate = 0.0015  
        self.current_step = 0
        self.cash = initial_cash
        self.shares = 0

    def reset(self) -> Observation:
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        return self.state()

    def state(self) -> Observation:
        # Get the current row of the dataset
        row = self.market_data.iloc[self.current_step]
        
        return Observation(
            cash=round(self.cash, 2),
            shares=self.shares,
            price=round(float(row['price']), 2),
            ema_20=round(float(row['ema_20']), 2),
            rsi_14=round(float(row['rsi_14']), 2),
            macd_histogram=round(float(row['macd_histogram']), 4),
            bb_lower=round(float(row['bb_lower']), 2),
            bb_upper=round(float(row['bb_upper']), 2),
            vwap=round(float(row['vwap']), 2)
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        current_price = float(self.market_data.iloc[self.current_step]['price'])
        prev_nav = self.cash + (self.shares * current_price)
        
        if action.action_type == "buy" and action.percentage > 0:
            investment = self.cash * action.percentage
            fee = investment * self.transaction_fee_rate
            shares_bought = int((investment - fee) / current_price)
            if shares_bought > 0:
                self.shares += shares_bought
                self.cash -= (shares_bought * current_price) + fee

        elif action.action_type == "sell" and action.percentage > 0:
            shares_to_sell = int(self.shares * action.percentage)
            if shares_to_sell > 0:
                gross_proceeds = shares_to_sell * current_price
                fee = gross_proceeds * self.transaction_fee_rate
                self.shares -= shares_to_sell
                self.cash += (gross_proceeds - fee)

        self.current_step += 1
        done = self.current_step >= self.total_steps - 1
        
        new_price = float(self.market_data.iloc[self.current_step]['price']) if not done else current_price
        current_nav = self.cash + (self.shares * new_price)
        
        reward_obj = Reward(
            step_reward=round(current_nav - prev_nav, 2),
            nav_change=round(current_nav - self.initial_cash, 2)
        )
        
        return self.state(), reward_obj, done, {}