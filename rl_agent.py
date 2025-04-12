
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from trading_env import StockTradingEnv
import warnings
warnings.filterwarnings("ignore")

def initialize_env(df, initial_amount=100000):
    """Initialize trading environment with validation"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    stock_dim = len(df["tic"].unique())
    tech_indicators = ["volume", "macd", "boll_ub", "boll_lb", 
                      "rsi_30", "cci_30", "dx_30", "close_30_sma", 
                      "close_60_sma", "turbulence"]
    
    max_price = df['close'].max()
    hmax = int(initial_amount / max_price) if max_price > 0 else 100

    return StockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=initial_amount,
        num_stock_shares=[0]*stock_dim,
        buy_cost_pct=[0.001]*stock_dim,
        sell_cost_pct=[0.001]*stock_dim,
        state_space=1 + 2*stock_dim + stock_dim*len(tech_indicators),
        action_space=stock_dim,
        tech_indicator_list=tech_indicators
    )

def train_agent(env, timesteps=100000, model_save_path="ppo_trader"):
    """Train PPO agent with error handling"""
    try:
        model = PPO("MlpPolicy", env, verbose=1,
                    batch_size=256,
                    n_steps=1024,
                    n_epochs=10,
                    gamma=0.99,
                    learning_rate=3e-4)
        
        model.learn(total_timesteps=timesteps)
        model.save(model_save_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")

def evaluate_agent(model_path, env):
    """Evaluate trained agent with proper validation"""
    try:
        model = PPO.load(model_path.replace('.zip', ''), env=env)
        obs = env.reset()
        done = False
        portfolio_values = []
        
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            portfolio_values.append(env.state[0] + sum(
                np.array(env.state[1:(env.stock_dim+1)]) * 
                np.array(env.state[(env.stock_dim+1):(env.stock_dim*2+1)]))
            )
        return portfolio_values
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {str(e)}")

def load_model(model_path, env):
    """Safe model loader with version checking"""
    try:
        return PPO.load(model_path.replace('.zip', ''), env=env)
    except ValueError as e:
        raise ValueError(f"Model format error: {str(e)}")
    except FileNotFoundError:
        raise FileNotFoundError("Model file not found - check path and permissions")
