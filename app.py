import streamlit as st
from rl_agent import train_rl_agent
from utils import load_data, preprocess_data

def main():
    st.title("Reinforcement Learning Stock Trading App")

    uploaded_file = st.file_uploader("Upload market data CSV file", type=["csv"])

    if uploaded_file:
        raw_data = load_data(uploaded_file)
        st.write("Raw Data:")
        st.dataframe(raw_data)

        processed_data = preprocess_data(raw_data)
        st.write("Processed Data:")
        st.dataframe(processed_data)

        if st.button("Train RL Agent"):
            train_rl_agent(processed_data)
            st.success("Training Completed!")

if __name__ == "__main__":
    main()
