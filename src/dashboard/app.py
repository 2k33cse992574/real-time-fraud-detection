import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# Configuration
API_URL = "http://api:8000/predict" # Docker service name

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Real-Time Credit Card Fraud Detection")
st.markdown("Enter transaction details below to get a real-time fraud probability prediction and SHAP explanation.")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar for inputs
st.sidebar.header("Transaction Features")
st.sidebar.markdown("Input the transaction properties.")

input_data = {}

input_data["amount"] = st.sidebar.number_input("Amount ($)", value=50.0, min_value=0.0)
input_data["transaction_hour"] = st.sidebar.slider("Transaction Hour", 0, 23, 12)
input_data["merchant_category"] = st.sidebar.selectbox(
    "Merchant Category", 
    ['Electronics', 'Travel', 'Grocery', 'Food', 'Clothing']
)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Indicators")
input_data["device_trust_score"] = st.sidebar.slider("Device Trust Score", 0, 100, 80)
input_data["velocity_last_24h"] = st.sidebar.number_input("Transactions in last 24h", value=1, min_value=0, step=1)
input_data["foreign_transaction"] = 1 if st.sidebar.checkbox("Foreign Transaction") else 0
input_data["location_mismatch"] = 1 if st.sidebar.checkbox("Location Mismatch") else 0

st.sidebar.markdown("---")
st.sidebar.subheader("Cardholder Info")
input_data["cardholder_age"] = st.sidebar.number_input("Cardholder Age", value=30, min_value=18, max_value=100)


if st.button("Predict Fraud Probability"):
    with st.spinner("Analyzing transaction..."):
        try:
            import os
            api_url = os.environ.get("API_URL", "http://api:8000/predict") 
            
            response = requests.post(api_url, json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                prob = result["fraud_probability"]
                is_fraud = result["is_fraud"]
                explanation = result["explanation"]
                
                # Update history
                st.session_state.history.insert(0, {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Amount ($)": f"{input_data['amount']:.2f}",
                    "Category": input_data['merchant_category'],
                    "Prob (%)": f"{prob*100:.1f}%",
                    "Status": "🚨 FRAUD" if is_fraud else "✅ LEGIT"
                })
                # Keep only last 5
                st.session_state.history = st.session_state.history[:5]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Result")
                    if is_fraud:
                        st.error("🚨 FRAUD DETECTED 🚨")
                    else:
                        st.success("✅ LEGITIMATE TRANSACTION")
                        
                    # Gauge Chart for Probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fraud Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if is_fraud else "darkgreen"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 100], 'color': "salmon"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)
                
                with col2:
                    st.subheader("SHAP Explanation (Why?)")
                    st.markdown("Features pushing the probability higher (red) vs lower (blue).")
                    
                    # Simple Waterfall chart using Plotly
                    feature_names = explanation["feature_names"]
                    shap_vals = explanation["shap_values"]
                    base_val = explanation["base_value"]
                    
                    # Sort by absolute SHAP value for better visualization (top 10)
                    sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:10]
                    
                    top_features = [feature_names[i] for i in sorted_indices]
                    top_shaps = [shap_vals[i] for i in sorted_indices]
                    
                    colors = ['red' if val > 0 else 'blue' for val in top_shaps]
                    
                    fig_shap = go.Figure(go.Bar(
                        x=top_shaps,
                        y=top_features,
                        orientation='h',
                        marker_color=colors
                    ))
                    
                    fig_shap.update_layout(
                        title="Top 10 Feature Contributions",
                        xaxis_title="SHAP Value (Impact on Model Output)",
                        yaxis_title="Feature",
                        yaxis={'categoryorder':'total ascending'}
                    )
                    
                    st.plotly_chart(fig_shap)
                    
                st.markdown("---")
                st.subheader("Recent Transactions History")
                
                history_df = pd.DataFrame(st.session_state.history)
                
                def color_status(val):
                    color = '#ff4b4b' if 'FRAUD' in val else '#00cc96'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(
                    history_df.style.map(color_status, subset=['Status']),
                    use_container_width=True,
                    hide_index=True
                )
                    
            else:
                st.error(f"Error from API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the API. Is it running?")
