import streamlit as st
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E88E5;
        font-weight: 500;
        margin-top: 2rem;
    }
    .section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        height: 100%;
    }
    .icon-large {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main page title with styled header
st.markdown("<h1 class='main-header'>Credit Card Fraud Detection System with Blockchain</h1>", unsafe_allow_html=True)

# Introduction section
st.markdown("""
This system combines machine learning with blockchain technology to create a transparent and immutable 
record of fraud detection results, providing an extra layer of security and auditability.
""")

# Create tabs for different sections of the about page
overview_tab, architecture_tab, models_tab, blockchain_tab, dataset_tab = st.tabs([
    "📋 Overview", 
    "🏗️ Architecture", 
    "🤖 Models", 
    "🔗 Blockchain", 
    "📊 Dataset"
])

with overview_tab:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>System Overview</h2>", unsafe_allow_html=True)
    
    # Overview description
    st.markdown("""
    ## About This System
    
    The Credit Card Fraud Detection System combines advanced machine learning algorithms with 
    blockchain technology to detect potentially fraudulent transactions while maintaining an 
    immutable record of all predictions.
    
    ### Key Objectives
    
    * Accurately identify fraudulent credit card transactions in real-time
    * Address the class imbalance problem inherent in fraud detection
    * Provide transparent and tamper-proof records of all fraud predictions
    * Enable detailed analysis of model performance and feature importance
    * Support model verification and auditability through blockchain integration
    """)
    
    # Create three columns for key features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("<div class='icon-large'>🔍</div>", unsafe_allow_html=True)
        st.markdown("<strong>Prediction Engine</strong>", unsafe_allow_html=True)
        st.markdown("""
        Submit transaction details to get real-time fraud predictions with confidence scores and risk analysis.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("<div class='icon-large'>📈</div>", unsafe_allow_html=True)
        st.markdown("<strong>Performance Analysis</strong>", unsafe_allow_html=True)
        st.markdown("""
        Compare multiple machine learning models with visualizations for accuracy, precision, recall, and ROC curves.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("<div class='icon-large'>⛓️</div>", unsafe_allow_html=True)
        st.markdown("<strong>Blockchain Explorer</strong>", unsafe_allow_html=True)
        st.markdown("""
        Access an immutable record of all fraud predictions on the Ethereum blockchain for transparency and auditability.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Navigation instructions
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("""
    ### Getting Started
    
    Use the sidebar navigation to access different pages:
    
    * **Prediction Page**: Submit transaction details for fraud analysis
    * **Performance Page**: Compare models and view detailed metrics
    
    The blockchain explorer is available through the frontend web interface.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with architecture_tab:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>System Architecture</h2>", unsafe_allow_html=True)
    
    # Architecture description
    st.markdown("""
    The system is built on a three-tier architecture that separates the frontend interface, 
    machine learning backend, and blockchain integration.
    
    ### Components
    
    1. **Frontend Layer**
       * Streamlit-based web interface for prediction and performance analysis
       * HTML/JavaScript blockchain explorer for transaction verification
    
    2. **Machine Learning Layer**
       * Multiple fraud detection models (Logistic Regression, Random Forest, XGBoost)
       * SMOTE implementation for addressing class imbalance
       * Feature engineering and preprocessing pipeline
    
    3. **Blockchain Layer**
       * Ethereum-based smart contracts for recording predictions
       * Immutable storage of model details and confidence scores
       * Verification mechanisms for model integrity
    """)
    
    # Create a simple architecture diagram using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define architecture boxes
    boxes = [
        dict(name="User Interface\n(Streamlit)", x=0.5, y=0.8, width=0.3, height=0.1, color='#C5E1A5'),
        
        dict(name="Prediction Engine", x=0.3, y=0.6, width=0.25, height=0.1, color='#90CAF9'),
        dict(name="Performance Analysis", x=0.7, y=0.6, width=0.25, height=0.1, color='#90CAF9'),
        
        dict(name="ML Models\n(LR, RF, XGB)", x=0.3, y=0.4, width=0.25, height=0.1, color='#FFCC80'),
        dict(name="SMOTE & Feature\nEngineering", x=0.7, y=0.4, width=0.25, height=0.1, color='#FFCC80'),
        
        dict(name="Smart Contracts\n(Ethereum)", x=0.5, y=0.2, width=0.4, height=0.1, color='#CE93D8'),
        
        dict(name="Blockchain", x=0.5, y=0.05, width=0.6, height=0.05, color='#F48FB1'),
    ]
    
    # Draw the boxes
    for box in boxes:
        ax.add_patch(plt.Rectangle(
            (box['x'] - box['width']/2, box['y'] - box['height']/2),
            box['width'], box['height'],
            linewidth=1, edgecolor='black', facecolor=box['color'], alpha=0.8
        ))
        ax.text(box['x'], box['y'], box['name'], ha='center', va='center', fontsize=10)
    
    # Add arrows
    ax.arrow(0.5, 0.75, 0, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.3, 0.55, 0, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.7, 0.55, 0, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.3, 0.35, 0.1, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.7, 0.35, -0.1, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.5, 0.15, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Set plot properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('System Architecture')
    
    # Display the diagram
    st.pyplot(fig)
    
    st.markdown("""
    ### Data Flow
    
    1. User submits transaction details via the Streamlit interface
    2. The prediction engine preprocesses data and applies the selected model
    3. Prediction results are displayed to the user and recorded on the blockchain
    4. Performance metrics are updated and available for analysis
    5. Blockchain explorer shows immutable record of predictions
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

with models_tab:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Machine Learning Models</h2>", unsafe_allow_html=True)
    
    # Models description
    st.markdown("""
    This system implements multiple machine learning models to provide a comprehensive approach to fraud detection.
    Each model offers different strengths and trade-offs in terms of accuracy, precision, recall, and computational efficiency.
    """)
    
    # Create a comparison table of models
    model_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Strengths": [
            "Fast training and inference, highly interpretable, lightweight",
            "Robust to outliers, captures non-linear relationships, less prone to overfitting",
            "State-of-the-art performance, handles complex feature interactions, high accuracy"
        ],
        "Use Case": [
            "Quick baseline model, when interpretability is critical",
            "General-purpose fraud detection with moderate complexity",
            "When maximum accuracy is required with sufficient computational resources"
        ],
        "Relative Performance": ["Good", "Better", "Best"]
    }
    
    model_df = pd.DataFrame(model_data)
    st.table(model_df)
    
    # SMOTE explanation
    st.markdown("""
    ### Handling Class Imbalance with SMOTE
    
    Fraud detection typically faces severe class imbalance, with fraudulent transactions representing 
    less than 1% of all transactions. This imbalance can lead to models that are biased toward the majority 
    class (legitimate transactions).
    
    **Synthetic Minority Over-sampling Technique (SMOTE)** addresses this challenge by:
    
    1. Creating synthetic examples of the minority class (fraudulent transactions)
    2. Generating these examples in feature space rather than simply duplicating existing instances
    3. Improving the model's ability to identify fraud patterns without simple memorization
    """)
    
    # Create a simple visualization of SMOTE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_minority = 10
    
    # Generate majority and minority samples
    X_majority = np.random.normal(loc=0, scale=1, size=(n_samples-n_minority, 2))
    X_minority = np.random.normal(loc=2, scale=0.5, size=(n_minority, 2))
    
    # Before SMOTE
    ax1.scatter(X_majority[:, 0], X_majority[:, 1], label='Legitimate', alpha=0.6, color='blue')
    ax1.scatter(X_minority[:, 0], X_minority[:, 1], label='Fraudulent', alpha=0.6, color='red')
    ax1.set_title('Before SMOTE')
    ax1.legend()
    
    # After SMOTE (simulated)
    # Generate synthetic minority samples
    n_synthetic = 30
    X_synthetic = np.random.normal(loc=2, scale=0.8, size=(n_synthetic, 2))
    
    ax2.scatter(X_majority[:, 0], X_majority[:, 1], label='Legitimate', alpha=0.6, color='blue')
    ax2.scatter(X_minority[:, 0], X_minority[:, 1], label='Fraudulent', alpha=0.6, color='red')
    ax2.scatter(X_synthetic[:, 0], X_synthetic[:, 1], label='Synthetic Fraud', alpha=0.4, color='orange')
    ax2.set_title('After SMOTE')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature importance explanation
    st.markdown("""
    ### Feature Importance
    
    The system analyzes and visualizes feature importance to provide insights into which transaction
    characteristics are most predictive of fraud. Common important features include:
    
    * Transaction amount (unusually large amounts often indicate fraud)
    * Time of day (many fraudulent transactions occur during off-hours)
    * Transaction category (certain merchant categories have higher fraud rates)
    * Geographic distance between customer and merchant
    * Customer transaction history and patterns
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

with blockchain_tab:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Blockchain Integration</h2>", unsafe_allow_html=True)
    
    # Blockchain description
    st.markdown("""
    This system leverages Ethereum blockchain technology to provide transparency, immutability, and 
    auditability for fraud detection predictions.
    
    ### Smart Contracts
    
    Two primary smart contracts manage the blockchain integration:
    
    1. **FraudDetection.sol**: Records fraud predictions with confidence scores and transaction data
    2. **ModelVerification.sol**: Stores and verifies machine learning models with performance metrics
    
    ### Benefits of Blockchain Integration
    
    * **Immutable Record**: Once recorded, fraud predictions cannot be altered or deleted
    * **Transparency**: All stakeholders can verify the history of predictions
    * **Auditability**: Facilitates compliance with regulatory requirements
    * **Model Governance**: Tracks model versions and their performance metrics
    * **Reduced Disputes**: Provides a single source of truth for fraud determinations
    """)
    
    # Show contract structs and functions
    st.markdown("""
    ### Key Smart Contract Structures
    
    **FraudDetection Contract:**
    ```solidity
    struct PredictionRecord {
        uint256 timestamp;       // When the prediction was made
        bool isFraud;            // Whether the transaction was classified as fraud
        uint256 confidence;      // Confidence score (0-100)
        string transactionData;  // Hashed transaction data
        address submittedBy;     // Address that submitted the prediction
    }
    ```
    
    **ModelVerification Contract:**
    ```solidity
    struct Model {
        string modelType;        // Type of model (Logistic Regression, Random Forest, XGBoost)
        string modelHash;        // Hash of the serialized model file
        string datasetHash;      // Hash of the dataset used for training
        string performanceMetrics; // JSON string with performance metrics
        uint256 timestamp;       // When the model was registered
        address registeredBy;    // Address that registered the model
        bool isActive;           // Whether this model is currently active
    }
    ```
    """)
    
    # Blockchain Explorer Overview
    st.markdown("""
    ### Blockchain Explorer
    
    The system includes a web-based blockchain explorer that allows users to:
    
    * View all recorded fraud predictions
    * Verify the confidence score and timestamp of each prediction
    * Check which model was used for a specific prediction
    * Monitor model performance metrics over time
    * Validate the integrity of models and predictions
    
    The explorer connects to an Ethereum node (local Ganache instance for development or public 
    Ethereum network for production) using Web3.js.
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

with dataset_tab:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
    
    # Dataset description
    st.markdown("""
    The system uses a credit card transaction dataset with labeled fraudulent and legitimate transactions.
    This enables supervised learning for fraud detection.
    """)
    
    # Check if dataset exists and display statistics
    try:
        df = pd.read_csv(r'C:\Users\chand\Documents\Fraud_Detection_System_with_Blockchain\sampled_dataset.csv')
        
        # Dataset statistics
        st.markdown("### Dataset Summary")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Transactions:** {len(df):,}")
            if 'is_fraud' in df.columns:
                fraud_count = df['is_fraud'].sum()
                legitimate_count = len(df) - fraud_count
                st.write(f"**Legitimate Transactions:** {legitimate_count:,} ({legitimate_count/len(df)*100:.2f}%)")
                st.write(f"**Fraudulent Transactions:** {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
        
        with col2:
            # Create a pie chart of the fraud distribution
            if 'is_fraud' in df.columns:
                fig, ax = plt.subplots(figsize=(5, 5))
                labels = ['Legitimate', 'Fraudulent']
                sizes = [legitimate_count, fraud_count]
                colors = ['#66BB6A', '#EF5350']
                explode = (0, 0.1)
                
                ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                       shadow=True, startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
        
        # Display dataset sample
        st.markdown("### Data Preview")
        st.write(df.head())
        
        # Display feature information
        st.markdown("### Feature Description")
        
        features = {
            "trans_date_trans_time": "Date and time of the transaction",
            "cc_num": "Credit card number (anonymized)",
            "merchant": "Merchant name where transaction occurred",
            "category": "Category of the purchase (e.g., grocery, entertainment)",
            "amt": "Transaction amount in USD",
            "first": "First name of the cardholder (anonymized)",
            "last": "Last name of the cardholder (anonymized)",
            "gender": "Gender of the cardholder (M/F)",
            "street": "Street address (anonymized)",
            "city": "City of the cardholder",
            "state": "State of the cardholder",
            "zip": "ZIP code of the cardholder",
            "lat": "Latitude of the cardholder's location",
            "long": "Longitude of the cardholder's location",
            "city_pop": "Population of the cardholder's city",
            "job": "Job of the cardholder",
            "dob": "Date of birth of the cardholder",
            "trans_num": "Transaction ID",
            "unix_time": "Unix timestamp of the transaction",
            "merch_lat": "Merchant latitude",
            "merch_long": "Merchant longitude",
            "is_fraud": "Target variable: 1 for fraudulent transaction, 0 for legitimate"
        }
        
        feature_df = pd.DataFrame({
            "Feature Name": features.keys(),
            "Description": features.values()
        })
        
        st.table(feature_df)
        
    except Exception as e:
        st.warning(f"Dataset preview not available: {e}")
        st.markdown("""
        ### Required Dataset Format
        
        The system expects a CSV file with transaction data including at minimum:
        
        * Transaction details (amount, date, merchant, category)
        * Cardholder information (anonymized for privacy)
        * Location data (optional but valuable for analysis)
        * Fraud label (is_fraud column with 1 for fraudulent transactions, 0 for legitimate)
        
        Sample datasets can be found on platforms like Kaggle or generated using simulation tools.
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar content
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the pages menu to navigate to:")
st.sidebar.markdown("- **Prediction Page**: Submit transactions for fraud analysis")
st.sidebar.markdown("- **Performance Page**: Compare model performance metrics")

# Blockchain connection status in sidebar
st.sidebar.title("Blockchain Status")
try:
    from web3 import Web3
    web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
    
    if web3.is_connected():
        st.sidebar.success("✅ Connected to Ethereum Network")
        
        # Check contract deployment
        try:
            with open('./frontend/src/contractAddress.json', 'r') as f:
                import json
                contract_data = json.load(f)
                contract_address = contract_data['FraudDetection']
                st.sidebar.info(f"Contract deployed at: {contract_address[:8]}...{contract_address[-6:]}")
        except:
            st.sidebar.warning("⚠️ Contract not deployed")
    else:
        st.sidebar.error("❌ Not connected to Ethereum Network")
        st.sidebar.markdown("To enable blockchain features:")
        st.sidebar.code("npx hardhat node\nnpx hardhat run scripts/deploy.js --network localhost")
except:
    st.sidebar.warning("⚠️ Web3 not available")
    st.sidebar.markdown("Install Web3: `pip install web3`")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Fraud Detection System with Blockchain Integration | Created with Streamlit, Scikit-Learn, and Ethereum</p>
    <p><small>For demonstration and educational purposes only</small></p>
</div>
""", unsafe_allow_html=True)