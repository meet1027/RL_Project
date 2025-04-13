import streamlit as st
from data_handler import DataHandler
from trading_env import StockTradingEnv
from rl_agent import RLAgent
from config import TICKERS, INDICATORS

def main():
    st.title("RL Stock Trading App")
    
    # Data Section
    with st.expander("Data Configuration"):
        uploaded_file = st.file_uploader("Upload market data")
        if uploaded_file:
            raw_df = pd.read_csv(uploaded_file)
            processed_df = DataHandler.preprocess_data(raw_df, INDICATORS)
            
    # Environment Setup
    with st.expander("Trading Environment"):
        if 'processed_df' in locals():
            env = StockTradingEnv(
                df=processed_df,
                stock_dim=len(TICKERS),
                hmax=100,
                initial_amount=100000,
                num_stock_shares=[0]*len(TICKERS),
                buy_cost_pct=[0.001]*len(TICKERS),
                sell_cost_pct=[0.001]*len(TICKERS),
                reward_scaling=1e-4,
                tech_indicator_list=INDICATORS
            )
            
    # Training Section
    if st.button("Start Training"):
        agent = RLAgent(env)
        agent.train()
        
if __name__ == "__main__":
    main()
