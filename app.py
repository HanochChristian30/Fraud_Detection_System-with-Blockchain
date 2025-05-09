import streamlit as st
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
from bayesian_utils import BayesianRiskCalculator

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
    .bayesian-highlight {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main page title with styled header
st.markdown("<h1 class='main-header'>Credit Card Fraud Detection System with Blockchain</h1>", unsafe_allow_html=True)

# Introduction section with Bayesian highlight
st.markdown("""
This system combines machine learning with blockchain technology to create a transparent and immutable 
record of fraud detection results, providing an extra layer of security and auditability.
""")

st.markdown("<div class='bayesian-highlight'>", unsafe_allow_html=True)
st.markdown("""
### NEW: Bayesian Risk Scoring
Our system now incorporates Bayesian risk scoring to enhance fraud detection accuracy. This approach:
- Updates fraud probabilities based on specific transaction risk factors
- Provides transparent explanations of risk indicators
- Combines statistical ML predictions with evidence-based Bayesian analysis
""")
st.markdown("</div>", unsafe_allow_html=True)

# Create tabs for different sections of the about page
overview_tab, architecture_tab, models_tab, bayesian_tab, blockchain_tab, dataset_tab = st.tabs([
    "📋 Overview", 
    "🏗️ Architecture", 
    "🤖 Models", 
    "📊 Bayesian Analysis",
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
    * Enhance explainability through Bayesian analysis of risk factors
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
       * Bayesian risk scoring for enhanced explainability
    
    3. **Blockchain Layer**
       * Ethereum-based smart contracts for recording predictions
       * Immutable storage of model details and confidence scores
       * Verification mechanisms for model integrity
    """)
    
    # Create a simple architecture diagram using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define architecture boxes
    boxes = [
        dict(name="User Interface\n(Streamlit)", x=0.5, y=0.9, width=0.3, height=0.1, color='#C5E1A5'),
        
        dict(name="Prediction Engine", x=0.3, y=0.7, width=0.25, height=0.1, color='#90CAF9'),
        dict(name="Performance Analysis", x=0.7, y=0.7, width=0.25, height=0.1, color='#90CAF9'),
        
        dict(name="ML Models\n(LR, RF, XGB)", x=0.2, y=0.5, width=0.2, height=0.1, color='#FFCC80'),
        dict(name="Bayesian Risk\nScoring", x=0.5, y=0.5, width=0.2, height=0.1, color='#FFAB91'),
        dict(name="SMOTE & Feature\nEngineering", x=0.8, y=0.5, width=0.2, height=0.1, color='#FFCC80'),
        
        dict(name="Smart Contracts\n(Ethereum)", x=0.5, y=0.3, width=0.4, height=0.1, color='#CE93D8'),
        
        dict(name="Blockchain", x=0.5, y=0.15, width=0.6, height=0.05, color='#F48FB1'),
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
    ax.arrow(0.5, 0.85, 0, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.3, 0.65, -0.05, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.3, 0.65, 0.15, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.7, 0.65, -0.15, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.7, 0.65, 0.05, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.2, 0.45, 0.25, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.5, 0.45, 0, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.8, 0.45, -0.25, -0.09, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax.arrow(0.5, 0.25, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
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
    3. Bayesian risk scoring analyzes transaction-specific risk factors
    4. A hybrid risk score combines ML model and Bayesian predictions
    5. Prediction results are displayed to the user and recorded on the blockchain
    6. Performance metrics are updated and available for analysis
    7. Blockchain explorer shows immutable record of predictions
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

with bayesian_tab:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Bayesian Risk Analysis</h2>", unsafe_allow_html=True)
    
    # Bayesian description
    st.markdown("""
    Our system now incorporates **Bayesian risk scoring** to enhance fraud detection accuracy and explainability.
    This statistical approach updates probabilities as new evidence becomes available, providing a transparent
    framework for fraud risk assessment.
    """)
    
    # How Bayesian works
    st.markdown("""
    ### How Bayesian Risk Scoring Works
    
    Bayesian risk scoring is based on Bayes' theorem, which states:
    
    $$P(Fraud|Evidence) = \\frac{P(Evidence|Fraud) \\times P(Fraud)}{P(Evidence)}$$
    
    Where:
    - P(Fraud|Evidence) is the **posterior probability** - what we want to calculate
    - P(Evidence|Fraud) is the **likelihood** - how likely we'd see this evidence if there is fraud
    - P(Fraud) is the **prior probability** - baseline fraud rate before seeing evidence
    - P(Evidence) is the total probability of the evidence
    
    In practical terms, the system:
    
    1. **Starts with prior probability** - The baseline fraud rate (typically 1-3%)
    2. **Calculates likelihood ratios** - How much more likely each risk factor is in fraudulent vs. legitimate transactions
    3. **Combines evidence** - Multiplies likelihood ratios to get a combined likelihood ratio
    4. **Updates probability** - Applies Bayes' theorem to get the final fraud probability
    """)
    
    # Display key risk factors
    st.markdown("""
    ### Key Risk Factors
    
    The system analyzes multiple risk factors, each with its own likelihood ratio:
    """)
    
    # Create a sample calculator to demonstrate
    calculator = BayesianRiskCalculator(prior_fraud_prob=0.01)
    
    # Create an interactive demonstration
    st.markdown("### Interactive Bayesian Risk Calculator")
    st.markdown("Adjust risk factors to see how they affect fraud probability:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_option = st.selectbox("Transaction Time", [
            "Business Hours (9 AM - 6 PM)", 
            "Evening (6 PM - 10 PM)", 
            "Late Night (10 PM - 6 AM)"
        ])
        
        amount_option = st.selectbox("Transaction Amount", [
            "Small (< $200)", 
            "Medium ($200 - $500)", 
            "Large ($500 - $1000)", 
            "Very Large (> $1000)"
        ])
        
        category_option = st.selectbox("Transaction Category", [
            "Low Risk (Grocery, Utilities, Healthcare)", 
            "Medium Risk (Clothing, Dining, Travel)", 
            "High Risk (Electronics, Jewelry, Gift Cards)"
        ])
    
    with col2:
        location_option = st.selectbox("Distance Between Customer & Merchant", [
            "Close (< 20 km)", 
            "Medium (20-50 km)", 
            "Far (50-100 km)", 
            "Very Far (> 100 km)"
        ])
        
        account_option = st.selectbox("Account Age", [
            "Established (> 90 days)", 
            "Recent (30-90 days)", 
            "New (7-30 days)", 
            "Very New (< 7 days)"
        ])
        
        device_option = st.selectbox("Device", [
            "Known Device", 
            "New Device"
        ])
    
    # Map selections to likelihood ratios
    time_lr = {
        "Business Hours (9 AM - 6 PM)": 1.0,
        "Evening (6 PM - 10 PM)": 1.5,
        "Late Night (10 PM - 6 AM)": 3.0
    }[time_option]
    
    amount_lr = {
        "Small (< $200)": 1.0,
        "Medium ($200 - $500)": 1.2,
        "Large ($500 - $1000)": 2.0,
        "Very Large (> $1000)": 5.0
    }[amount_option]
    
    category_lr = {
        "Low Risk (Grocery, Utilities, Healthcare)": 1.0,
        "Medium Risk (Clothing, Dining, Travel)": 2.0,
        "High Risk (Electronics, Jewelry, Gift Cards)": 4.0
    }[category_option]
    
    location_lr = {
        "Close (< 20 km)": 1.0,
        "Medium (20-50 km)": 2.0,
        "Far (50-100 km)": 4.0,
        "Very Far (> 100 km)": 8.0
    }[location_option]
    
    account_lr = {
        "Established (> 90 days)": 1.0,
        "Recent (30-90 days)": 1.3,
        "New (7-30 days)": 2.0,
        "Very New (< 7 days)": 5.0
    }[account_option]
    
    device_lr = {
        "Known Device": 1.0,
        "New Device": 6.0
    }[device_option]
    
    # Calculate combined likelihood ratio
    combined_lr = time_lr * amount_lr * category_lr * location_lr * account_lr * device_lr
    
    # Calculate posterior probability
    prior_prob = 0.01  # 1% prior probability of fraud
    prior_odds = prior_prob / (1 - prior_prob)
    posterior_odds = prior_odds * combined_lr
    posterior_prob = posterior_odds / (1 + posterior_odds)
    
    # Display results
    st.markdown("### Bayesian Analysis Results")
    
    # Risk gauge using matplotlib
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create a horizontal gauge
    ax.barh([0], [1], color='lightgrey', height=0.3)
    ax.barh([0], [posterior_prob], color='#FF5252' if posterior_prob > 0.5 else '#FFA726' if posterior_prob > 0.1 else '#66BB6A', height=0.3)
    
    # Add markers for different risk levels
    ax.axvline(x=0.1, color='#66BB6A', linestyle='--', alpha=0.7)
    ax.axvline(x=0.5, color='#FFA726', linestyle='--', alpha=0.7)
    ax.axvline(x=0.8, color='#FF5252', linestyle='--', alpha=0.7)
    
    # Add labels
    ax.text(0.05, -0.5, "Low Risk", ha='center', va='center')
    ax.text(0.3, -0.5, "Medium Risk", ha='center', va='center')
    ax.text(0.65, -0.5, "High Risk", ha='center', va='center')
    ax.text(0.9, -0.5, "Very High Risk", ha='center', va='center')
    
    # Add fraud probability as percentage
    ax.text(posterior_prob, 0, f" {posterior_prob:.1%}", va='center')
    
    # Set plot limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    ax.set_title('Fraud Probability')
    
    st.pyplot(fig)
    
    # Display the calculation details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Probability Breakdown")
        st.markdown(f"**Prior Probability:** {prior_prob:.2%}")
        st.markdown(f"**Combined Likelihood Ratio:** {combined_lr:.2f}x")
        st.markdown(f"**Posterior Probability:** {posterior_prob:.2%}")
        
        risk_label = "HIGH" if posterior_prob > 0.5 else "MEDIUM" if posterior_prob > 0.1 else "LOW"
        risk_color = "red" if posterior_prob > 0.5 else "orange" if posterior_prob > 0.1 else "green"
        st.markdown(f"**Risk Assessment:** <span style='color:{risk_color};font-weight:bold;'>{risk_label}</span>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Individual Risk Factors")
        
        # Create a DataFrame for the likelihood ratios
        lr_data = pd.DataFrame({
            'Factor': ['Time of Day', 'Transaction Amount', 'Merchant Category', 
                      'Location', 'Account Age', 'Device'],
            'Likelihood Ratio': [time_lr, amount_lr, category_lr, 
                                location_lr, account_lr, device_lr]
        })
        
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        y_pos = np.arange(len(lr_data['Factor']))
        ax.barh(y_pos, lr_data['Likelihood Ratio'], color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(lr_data['Factor'])
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Likelihood Ratio')
        ax.set_title('Risk Factors Impact')
        
        # Add a vertical line at x=1 (baseline)
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(lr_data['Likelihood Ratio']):
            ax.text(v + 0.1, i, f'{v:.1f}x', va='center')
        
        st.pyplot(fig)
    
    # Advantages of Bayesian approach
    st.markdown("""
    ### Advantages of Bayesian Risk Scoring
    
    1. **Transparent and Explainable**: Each risk factor's contribution is clearly visible
    2. **Combines Domain Knowledge with Data**: Incorporates expert insights with statistical patterns
    3. **Handles Uncertainty**: Provides probabilities rather than binary decisions
    4. **Adapts to New Evidence**: Naturally updates as more information becomes available
    5. **Works with Limited Data**: Effective even without large training datasets
    
    Our system combines this Bayesian approach with machine learning models for a robust hybrid fraud detection system.
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

    # Add a screenshot mockup of the blockchain explorer
    st.image("https://i.ibb.co/wN7rQb7/blockchain-explorer-mockup.png", 
             caption="Blockchain Explorer Interface (Mockup)", 
             use_column_width=True)
    
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

# Add Bayesian risk calculator in sidebar
st.sidebar.title("Quick Risk Calculator")
amount = st.sidebar.slider("Transaction Amount ($)", 1, 2000, 100)
time_options = {
    "Morning (6 AM - 12 PM)": 1.0,
    "Afternoon (12 PM - 6 PM)": 1.0,
    "Evening (6 PM - 10 PM)": 1.5,
    "Late Night (10 PM - 6 AM)": 3.0
}
time_of_day = st.sidebar.selectbox("Time of Day", list(time_options.keys()))
category_options = {
    "Grocery": 1.0,
    "Restaurant": 1.2,
    "Retail": 1.5,
    "Gas": 1.0,
    "Travel": 2.0,
    "Electronics": 3.5,
    "Jewelry": 4.0
}
category = st.sidebar.selectbox("Category", list(category_options.keys()))

# Calculate quick risk score
quick_calc = BayesianRiskCalculator(prior_fraud_prob=0.01)
amount_lr = 5.0 if amount > 1000 else 2.0 if amount > 500 else 1.2 if amount > 200 else 1.0
time_lr = time_options[time_of_day]
category_lr = category_options[category]

combined_lr = amount_lr * time_lr * category_lr
prior_odds = 0.01 / 0.99
posterior_odds = prior_odds * combined_lr
posterior_prob = posterior_odds / (1 + posterior_odds)

# Show quick risk assessment
risk_color = "red" if posterior_prob > 0.5 else "orange" if posterior_prob > 0.1 else "green"
st.sidebar.markdown(f"### Quick Risk Assessment")
st.sidebar.markdown(f"Risk Score: <span style='color:{risk_color};font-weight:bold;'>{posterior_prob:.1%}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"Amount Risk: {amount_lr:.1f}x")
st.sidebar.markdown(f"Time Risk: {time_lr:.1f}x")
st.sidebar.markdown(f"Category Risk: {category_lr:.1f}x")
st.sidebar.markdown("*This is a simplified calculation. Use the Prediction page for comprehensive analysis.*")

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
    <p><small>Enhanced with Bayesian Risk Scoring for improved explainability</small></p>
</div>
""", unsafe_allow_html=True)