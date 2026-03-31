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
    def __init__(self, market_data: pd.DataFrame):
        super().__init__("Capital Preservation", "Easy", market_data)

    def grade(self, final_nav: float) -> float:
        nav_min = self.initial_cash * 0.90  
        nav_max = self.initial_cash         
        score = (final_nav - nav_min) / (nav_max - nav_min)
        return max(0.0, min(1.0, score))

class AlphaGenerationTask(TradingTask):
    def __init__(self, market_data: pd.DataFrame):
        super().__init__("Alpha Generation", "Medium", market_data)

    def grade(self, final_nav: float) -> float:
        # Extract the first and last prices from the 'price' column of the DataFrame
        starting_price = self.market_data.iloc[0]['price']
        ending_price = self.market_data.iloc[-1]['price']
        
        shares_bought = self.initial_cash / starting_price
        buy_and_hold_nav = shares_bought * ending_price
        
        nav_min = buy_and_hold_nav               
        nav_max = buy_and_hold_nav * 1.15        
        
        score = (final_nav - nav_min) / (nav_max - nav_min)
        return max(0.0, min(1.0, score))

class BearMarketTask(TradingTask):
    def __init__(self, market_data: pd.DataFrame):
        super().__init__("Bear Market Navigation", "Hard", market_data)

    def grade(self, final_nav: float) -> float:
        nav_min = self.initial_cash * 0.95  
        nav_max = self.initial_cash * 1.10  
        score = (final_nav - nav_min) / (nav_max - nav_min)
        return max(0.0, min(1.0, score))