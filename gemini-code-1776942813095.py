import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Page config
st.set_page_config(page_title="Bank Profitability Analysis", layout="wide")

@st.cache_data
def load_data():
    # Loading the local file as requested
    df = pd.read_csv("Indian_Banks.csv", encoding='latin1')
    
    # Cleaning column names (handling newlines found in the CSV)
    df.columns = [c.replace('\n', ' ') for c in df.columns]
    
    # Preprocessing logic from notebook
    df = df.fillna(df.mean(numeric_only=True))
    
    # Feature Engineering
    df['Log_Assets'] = df['Bank Size [Log(Assets)]']
    df['NPA_CAR'] = df['Net NPA Ratio (%)'] * df['CAR / CRAR (%)']
    df['NPA_Size'] = df['Net NPA Ratio (%)'] * df['Log_Assets']
    
    return df

def train_model(df):
    # Defining features based on your notebook logic
    features = ['Net NPA Ratio (%)', 'CAR / CRAR (%)', 'Credit Growth (%)', 
                'Cost-to- Income Ratio (%)', 'Log_Assets', 'NPA_CAR', 'NPA_Size']
    target = 'ROA (%) [DV]'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred)
    }
    
    return model, metrics, features

# App UI
st.title("📊 NPA Impact on Bank Profitability")
st.markdown("This app analyzes how Non-Performing Assets (NPA) and other financial ratios impact the Return on Assets (ROA) of Indian Banks.")

try:
    df = load_data()
    model, metrics, feature_list = train_model(df)

    # Sidebar - Predictor
    st.sidebar.header("Predict ROA")
    input_data = {}
    for feat in feature_list:
        if "Interaction" not in feat: # Let the app calculate interactions
            val = st.sidebar.number_input(f"Enter {feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))
            input_data[feat] = val

    # Calculate interactions for the model
    input_data['NPA_CAR'] = input_data['Net NPA Ratio (%)'] * input_data['CAR / CRAR (%)']
    input_data['NPA_Size'] = input_data['Net NPA Ratio (%)'] * input_data['Log_Assets']
    
    input_df = pd.DataFrame([input_data])[feature_list]
    prediction = model.predict(input_df)[0]

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Performance")
        st.write(f"**Random Forest R² Score:** {metrics['R2']:.4f}")
        st.write(f"**Mean Squared Error:** {metrics['MSE']:.4f}")
        
        st.metric("Predicted ROA (%)", f"{prediction:.2f}%")

    with col2:
        st.subheader("ROA Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['ROA (%) [DV]'], kde=True, ax=ax)
        st.pyplot(fig)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))

except FileNotFoundError:
    st.error("Error: 'Indian_Banks.csv' not found. Please ensure it is in the repository.")