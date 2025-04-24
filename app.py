import streamlit as st
import os
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Main page content
st.markdown("""
# <span style='color:#FF4B4B'>Credit Card Fraud Detection System</span>
""", unsafe_allow_html=True)
st.write("Welcome to the Credit Card Fraud Detection System. Use the sidebar to navigate to different pages.")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Home", "Prediction", "Performance"]
selected_page = st.sidebar.radio("Go to", pages)

# Check if we're on the home page
if selected_page == "Home":
    # Display information about blockchain integration
    st.subheader("Blockchain-Powered Fraud Detection")
    st.write("""
    This system integrates blockchain technology to securely record fraud predictions, 
    ensuring transparency and immutability of detection results. Each prediction is 
    stored on the Ethereum blockchain with a confidence score and transaction hash.
    """)
    
    # Display instructions
    st.subheader("Instructions")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Prediction Page:**
        - Enter transaction details
        - Select machine learning model
        - Configure risk threshold
        - Get fraud predictions with risk analysis
        - View blockchain transaction record
        """)
    
    with col2:
        st.markdown("""
        **Performance Page:**
        - Compare model metrics
        - View precision-recall curves
        - Analyze feature importance
        - Evaluate threshold impact
        """)

    # Section explaining the technical aspects
    st.subheader("Technical Overview")
    st.write("""
    This fraud detection system uses advanced machine learning to identify potentially fraudulent credit card transactions.
    It incorporates SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance, a common challenge
    in fraud detection where fraudulent transactions are significantly less common than legitimate ones.
    """)

    # Create two columns for key features and models
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Key Features:**")
        st.markdown("""
        - 📊 Multiple machine learning models
        - ⚖️ SMOTE implementation for class balancing
        - 🔒 Blockchain integration for secure record-keeping
        - 📈 Detailed performance metrics and visualizations
        - ⚡ Real-time transaction risk assessment
        - 🌐 Location-based risk analysis
        """)
    
    with col4:
        st.markdown("**Models Available:**")
        st.markdown("""
        - **Logistic Regression**: Fast baseline model with interpretable results
        - **Random Forest**: Robust ensemble method less prone to overfitting
        - **XGBoost**: High-performance gradient boosting for maximum accuracy
        """)

    # Display sample dataset section
    st.subheader("Dataset Preview")
    
    # Check if dataset exists and display preview
    try:
        df = pd.read_csv(r'C:\Users\chand\Documents\Fraud_Detection_System_with_Blockchain\sampled_dataset.csv')
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
        st.warning("Sample dataset not found. Upload a dataset to view preview.")
        
    # Model storage information
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if os.path.exists(models_path):
        model_files = [f for f in os.listdir(models_path) if f.endswith('.pkl')]
        if model_files:
            st.success(f"✅ {len(model_files)} pre-trained models available: {', '.join(model_files)}")
        else:
            st.warning("No pre-trained models found. Train models on the Prediction page.")
    else:
        st.warning("Models directory not found. Models will be created when training.")

elif selected_page == "Prediction":
    st.info("Please navigate to the Prediction page from the application menu.")
    
elif selected_page == "Performance":
    st.info("Please navigate to the Performance page from the application menu.")

# Footer
st.markdown("---")
st.markdown("""
**Note**: For optimal performance, ensure your Ethereum node is running if you want to use blockchain features.  
*This application is for demonstration purposes only.*
""")