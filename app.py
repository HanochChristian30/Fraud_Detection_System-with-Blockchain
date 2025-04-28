import streamlit as st
import os
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Main page content
st.markdown("""
# <span style='color:#FF4B4B'>Credit Card Fraud Detection System</span>
""", unsafe_allow_html=True)

# Display information about blockchain integration
st.subheader("Blockchain-Powered Fraud Detection")
st.write("""
This system integrates blockchain technology to securely record fraud predictions, 
ensuring transparency and immutability of detection results. Each prediction is 
stored on the Ethereum blockchain with a confidence score and transaction hash.
""")

# Create tabs for main functionality
tab1, tab2 = st.tabs(["Prediction", "Performance"])

with tab1:
    st.markdown("## Transaction Prediction")
    st.write("Enter transaction details to predict if it's fraudulent.")
    
    # Import and display the prediction interface
    try:
        from pages.Prediction import display_prediction_ui
        display_prediction_ui()
    except ImportError:
        st.error("Prediction module not found. Please make sure the files are properly organized.")
        st.info("You can navigate to the Prediction.py file directly to use prediction functionality.")

with tab2:
    st.markdown("## Model Performance")
    st.write("Compare performance metrics across different fraud detection models.")
    
    # Import and display the performance interface
    try:
        from pages.performance import display_performance_ui
        display_performance_ui()
    except ImportError:
        st.error("Performance module not found. Please make sure the files are properly organized.")
        st.info("You can navigate to the performance.py file directly to view model performance.")

# Display sample dataset section
st.subheader("Dataset Preview")

# Check if dataset exists and display preview
try:
    df = pd.read_csv(r'C:\Users\chand\Documents\Fraud_Detection_System_with_Blockchain\sampled_dataset.csv')
    
    with st.expander("View Sample Data"):
        st.write(df.head())
        
        # Display fraud distribution
        if 'is_fraud' in df.columns:
            fraud_count = df['is_fraud'].sum()
            legitimate_count = len(df) - fraud_count
            
            st.markdown(f"**Dataset contains:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"✓ {legitimate_count} legitimate transactions")
            with col_b:
                st.warning(f"⚠️ {fraud_count} fraudulent transactions ({fraud_count/len(df)*100:.2f}%)")
except Exception as e:
    st.warning("Sample dataset not found. Please upload a dataset to view preview.")

# Footer
st.markdown("---")
st.markdown("""
**Note**: For optimal performance, ensure your Ethereum node is running if you want to use blockchain features.  
*This application is for demonstration purposes only.*
""")