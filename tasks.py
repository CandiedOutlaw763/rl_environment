import pandas as pd

class TradingTask:
    def __init__(self, name: str, difficulty: str, market_data: pd.DataFrame, initial_cash: float = 10000.0):
        self.name = name
        self.difficulty = difficulty
        self.market_data = market_data.reset_index(drop=True)
        self.initial_cash = initial_cash

    def grade(self, final_nav: float) -> float:
        raise NotImplementedError("Each task must implement its own grading logic.")

class CapitalPreservationTask(TradingTask):
    """Easy Task: Sideways market. Survive with minimal drawdown, small profits are rewarded."""
    def __init__(self, market_data: pd.DataFrame):
        super().__init__("Capital Preservation", "Easy", market_data)

    def grade(self, final_nav: float) -> float:
        nav_min = self.initial_cash * 0.98  # 0.0 if you lose 2% or more
        nav_max = self.initial_cash * 1.02  # 1.0 if you make 2% or more
        
        score = (final_nav - nav_min) / (nav_max - nav_min)
        return max(0.0, min(1.0, score))

class AlphaGenerationTask(TradingTask):
    """Medium Task: Bull market. You MUST beat Buy and Hold to score points."""
    def __init__(self, market_data: pd.DataFrame):
        super().__init__("Alpha Generation", "Hard", market_data)

    def grade(self, final_nav: float) -> float:
        starting_price = self.market_data.iloc[0]['price']
        ending_price = self.market_data.iloc[-1]['price']
        
        buy_and_hold_nav = (self.initial_cash / starting_price) * ending_price
        
        nav_min = buy_and_hold_nav          # 0.0 if you just match or underperform the market
        nav_max = buy_and_hold_nav * 1.02   # 1.0 if you beat the market by 2% through active trading
        
        score = (final_nav - nav_min) / (nav_max - nav_min)
        return max(0.0, min(1.0, score))

class BearMarketTask(TradingTask):
    """Hard Task: Bear market. Protect capital better than Buy and Hold."""
    def __init__(self, market_data: pd.DataFrame):
        super().__init__("Bear Market Navigation", "Medium", market_data)

    def grade(self, final_nav: float) -> float:
        starting_price = self.market_data.iloc[0]['price']
        ending_price = self.market_data.iloc[-1]['price']
        
        buy_and_hold_nav = (self.initial_cash / starting_price) * ending_price
        
        nav_min = buy_and_hold_nav          # 0.0 if you lose as much as the market
        nav_max = self.initial_cash         # 1.0 if you protect your capital entirely
        
        score = (final_nav - nav_min) / (nav_max - nav_min)
        return max(0.0, min(1.0, score))