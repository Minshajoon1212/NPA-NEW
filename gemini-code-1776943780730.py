import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Bank Profitability Analysis", layout="wide")

@st.cache_data
def load_and_clean_data():
    # Load the local CSV file
    df = pd.read_csv("Indian_Banks.csv", encoding='latin1')
    
    # Cleaning column names to remove hidden newlines/special characters
    df.columns = [c.replace('\n', ' ').strip() for c in df.columns]
    
    # Filling missing values as per notebook logic
    df = df.fillna(df.mean(numeric_only=True))
    
    # Feature engineering for the model
    df['Log_Assets'] = df['Bank Size [Log(Assets)]']
    df['NPA_CAR'] = df['Net NPA Ratio (%)'] * df['CAR / CRAR (%)']
    df['NPA_Size'] = df['Net NPA Ratio (%)'] * df['Log_Assets']
    
    return df

def run_analysis():
    st.title("📈 NPA Impact on Bank Profitability")
    
    try:
        df = load_and_clean_data()
        
        # Display Data
        if st.checkbox("Show Raw Data"):
            st.write(df.head())

        # Visualization Section (Uses Matplotlib)
        st.subheader("Distribution of ROA")
        fig, ax = plt.subplots()
        ax.hist(df['ROA (%) [DV]'], bins=20, color='skyblue', edgecolor='black')
        ax.set_title("ROA Distribution")
        ax.set_xlabel("ROA (%)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("Financial Ratios Correlation")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

    except FileNotFoundError:
        st.error("Indian_Banks.csv not found in the root folder.")

if __name__ == "__main__":
    run_analysis()