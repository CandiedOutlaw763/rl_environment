from pydantic import BaseModel, Field
from typing import Literal

class Observation(BaseModel):
    cash: float = Field(description="Current available cash balance")
    shares: int = Field(description="Number of asset shares currently held")
    price: float = Field(description="The current market price of the asset")
    
    # Technical Indicators
    ema_20: float = Field(description="20-period Exponential Moving Average (Trend)")
    rsi_14: float = Field(description="14-period Relative Strength Index (Momentum)")
    macd_histogram: float = Field(description="MACD Histogram value (Momentum)")
    bb_lower: float = Field(description="Bollinger Bands Lower Band (Volatility)")
    bb_upper: float = Field(description="Bollinger Bands Upper Band (Volatility)")
    vwap: float = Field(description="Volume Weighted Average Price (Volume/Trend)")

class Action(BaseModel):
    action_type: Literal["buy", "sell", "hold"] = Field(description="The type of trade to execute")
    percentage: float = Field(ge=0.0, le=1.0, description="Percentage of cash/shares to trade (0.0 to 1.0)")

class Reward(BaseModel):
    step_reward: float = Field(description="Immediate change in Net Asset Value (NAV)")
    nav_change: float = Field(description="Total change in NAV since the start")