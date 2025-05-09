import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import datetime
import os
import sys
# Add the project root to the path to ensure bayesian_utils can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bayesian_utils import BayesianRiskCalculator
from imblearn.over_sampling import SMOTE
import pickle
import time
import json
import hashlib
from web3 import Web3
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Fraud Detection - Prediction", layout="wide")

# Page title with better styling
st.markdown("""
# <span style='color:#FF4B4B'>Fraud Detection System</span>
""", unsafe_allow_html=True)
st.write("Enter transaction details to predict if it's fraudulent.")

# Blockchain Connection Setup
st.sidebar.title("Blockchain Connection")
# Connect to blockchain
try:
    web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))  # Ganache default URL
    
    # Check if connected
    if web3.is_connected():
        st.sidebar.success("Connected to Ethereum network")
        
        # Load contract address
        try:
            with open('./frontend/src/contractAddress.json', 'r') as f:
                contract_data = json.load(f)
                contract_address = contract_data['FraudDetection']
        except Exception as e:
            st.sidebar.warning(f"Error loading contract address: {e}")
            contract_address = None

        # Load contract ABI
        try:
            with open('./frontend/src/artifacts/contracts/FraudDetection.sol/FraudDetection.json', 'r') as f:
                contract_json = json.load(f)
                contract_abi = contract_json['abi']
        except Exception as e:
            st.sidebar.warning(f"Error loading contract ABI: {e}")
            contract_abi = None

        # Initialize contract
        if contract_address and contract_abi:
            contract = web3.eth.contract(address=contract_address, abi=contract_abi)
            
            # Get accounts
            accounts = web3.eth.accounts
            if accounts:
                account = st.sidebar.selectbox("Select Account", accounts)
                st.sidebar.write(f"Account: {account}")
                st.sidebar.write(f"Balance: {web3.from_wei(web3.eth.get_balance(account), 'ether')} ETH")
            else:
                st.sidebar.error("No accounts found")
                account = None
                contract = None
        else:
            st.sidebar.warning("Contract not properly configured")
            account = None
            contract = None
    else:
        st.sidebar.error("Not connected to Ethereum network")
        web3 = None
        account = None
        contract = None
except Exception as e:
    st.sidebar.error(f"Blockchain connection error: {e}")
    web3 = None
    account = None
    contract = None

# Function to load data with proper caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\chand\Documents\Fraud_Detection_System_with_Blockchain\sampled_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure the file exists at the specified path.")
        st.stop()

# Preprocess data for the specific dataset structure
def preprocess_data(df):
    processed_df = df.copy()
    
    # Handle date-time features
    if 'trans_date_trans_time' in processed_df.columns:
        # Support both date formats by setting dayfirst=True
        processed_df['trans_date_trans_time'] = pd.to_datetime(processed_df['trans_date_trans_time'], errors='coerce', dayfirst=True)
        processed_df['hour'] = processed_df['trans_date_trans_time'].dt.hour
        processed_df['day'] = processed_df['trans_date_trans_time'].dt.day
        processed_df['month'] = processed_df['trans_date_trans_time'].dt.month
        processed_df['dayofweek'] = processed_df['trans_date_trans_time'].dt.dayofweek
        processed_df.drop(['trans_date_trans_time'], axis=1, inplace=True)
    
    # Convert gender to numeric
    if 'gender' in processed_df.columns:
        processed_df['gender'] = processed_df['gender'].map({'M': 1, 'F': 0})
    
    # Handle date of birth - Fix the date format issue
    if 'dob' in processed_df.columns:
        # Support both date formats by setting dayfirst=True
        processed_df['dob'] = pd.to_datetime(processed_df['dob'], errors='coerce', dayfirst=True)
        current_year = pd.Timestamp.now().year
        processed_df['age'] = current_year - processed_df['dob'].dt.year
        processed_df.drop(['dob'], axis=1, inplace=True)
    
    # Drop irrelevant or personally identifiable columns
    columns_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 
                        'state', 'zip', 'trans_num', 'unix_time']
    columns_to_drop = [col for col in columns_to_drop if col in processed_df.columns]
    processed_df.drop(columns_to_drop, axis=1, inplace=True)
    
    # One-hot encode categorical columns
    categorical_columns = ['category', 'merchant', 'job']
    categorical_columns = [col for col in categorical_columns if col in processed_df.columns]
    
    if categorical_columns:
        processed_df = pd.get_dummies(processed_df, columns=categorical_columns, drop_first=True)
    
    # Rename columns for consistency
    if 'amt' in processed_df.columns:
        processed_df.rename(columns={'amt': 'amount'}, inplace=True)
    
    # Handle NaN values - Add imputation step
    numerical_features = processed_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = processed_df.select_dtypes(include=['object', 'category']).columns
    
    # Impute numerical features with median
    if len(numerical_features) > 0:
        num_imputer = SimpleImputer(strategy='median')
        processed_df[numerical_features] = num_imputer.fit_transform(processed_df[numerical_features])
    
    # Impute categorical features with most frequent value
    if len(categorical_features) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        processed_df[categorical_features] = cat_imputer.fit_transform(processed_df[categorical_features])
    
    return processed_df

# Load data
df = load_data()

# Let user inspect the data
with st.expander("Preview Dataset"):
    st.write(df.head())
    # Display fraud distribution
    if 'is_fraud' in df.columns:
        fraud_count = df['is_fraud'].sum()
        legitimate_count = len(df) - fraud_count
        st.write(f"Dataset contains {legitimate_count} legitimate and {fraud_count} fraudulent transactions")
        st.write(f"Fraud rate: {fraud_count/len(df)*100:.2f}%")

# Check if required columns exist
required_columns = ['is_fraud']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

# Check if saved models exist and create a checkbox for using them
models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Define model paths
lr_path = os.path.join(models_path, "logistic_regression.pkl")
rf_path = os.path.join(models_path, "random_forest.pkl")
xgb_path = os.path.join(models_path, "xgboost.pkl")

# Check if models exist
models_exist = (os.path.exists(lr_path) and os.path.exists(rf_path) and os.path.exists(xgb_path))

# Create additional settings in sidebar
st.sidebar.title("Settings")

# Option to use saved models - FIXED: Initialize variable before using
use_saved_models = st.sidebar.checkbox("Use pre-trained models", value=models_exist, disabled=not models_exist)
if not models_exist and use_saved_models:
    st.sidebar.warning("No saved models found. Will train new models.")

# Model selection in sidebar
model_choice = st.sidebar.selectbox("Select Model", 
                                ['Logistic Regression', 'Random Forest', 'XGBoost'],
                                help="Choose which model to use for prediction")

# Add SMOTE option to sidebar with improved warning
use_smote = st.sidebar.checkbox("Apply SMOTE for training", value=False, 
                            help="Use Synthetic Minority Over-sampling to address class imbalance (may fail with very large datasets)")

# Risk threshold adjustment
risk_threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5, 
                                help="Adjust threshold for classifying a transaction as fraudulent")

# Bayesian weighting slider
bayesian_weight = st.sidebar.slider("Bayesian Weight", 0.0, 1.0, 0.4, 
                                 help="Weight given to Bayesian analysis (vs ML model) in final risk score")

# Add "Train New Models" button
if st.sidebar.button("Train & Save Models"):
    with st.spinner("Training and saving all models..."):
        # Process the dataset - Fixed to handle larger datasets better
        processed_df = preprocess_data(df)
        
        # Train-test split
        X = processed_df.drop('is_fraud', axis=1)
        y = processed_df['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Apply SMOTE if selected - Improved with better error handling
        if use_smote:
            try:
                # Check if dataset isn't too large for SMOTE
                if len(X_train) > 100000:  # Arbitrary threshold for "large" dataset
                    st.sidebar.warning("Dataset too large for SMOTE. Training without SMOTE.")
                    use_smote_actual = False
                else:
                    # Make sure we have no NaN values before applying SMOTE
                    if X_train.isna().any().any():
                        st.sidebar.warning("Data contains NaN values. Applying imputation before SMOTE.")
                        # Impute missing values
                        num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
                        num_imputer = SimpleImputer(strategy='median')
                        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
                        
                    smote = SMOTE(sampling_strategy=0.5, k_neighbors=min(5, sum(y_train)-1), random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    st.sidebar.success(f"Applied SMOTE: Balanced training data")
                    use_smote_actual = True
            except Exception as e:
                st.sidebar.warning(f"SMOTE could not be applied: {str(e)}")
                use_smote_actual = False
        else:
            use_smote_actual = False
        
        # Train all models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight=None if use_smote_actual else 'balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight=None if use_smote_actual else 'balanced'),
            'XGBoost': XGBClassifier(eval_metric='logloss', scale_pos_weight=1 if use_smote_actual else y_train.value_counts()[0]/y_train.value_counts()[1])
        }
        
        # Train and save all models
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            # Save model
            model_filename = name.lower().replace(" ", "_") + ".pkl"
            model_path = os.path.join(models_path, model_filename)
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
        
        st.sidebar.success("All models trained and saved successfully!")
        # Set flag to use saved models
        use_saved_models = True

# Create two columns for the main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transaction Details")
    
    # Date and time
    today = datetime.datetime.now()
    transaction_date = st.date_input("Transaction Date", today)
    transaction_time = st.time_input("Transaction Time", datetime.time(12, 0))
    
    # Extract unique merchants and categories from dataset for proper selection
    if 'merchant' in df.columns:
        merchants = sorted(df['merchant'].unique().tolist())
        merchant = st.selectbox("Merchant", merchants)
    else:
        merchant = st.text_input("Merchant", "Unknown")
        
    if 'category' in df.columns:
        categories = sorted(df['category'].unique().tolist())
        category = st.selectbox("Category", categories)
    else:
        category = st.text_input("Category", "retail")
    
    # Amount with more realistic defaults
    amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=120.0)
    
    # Transaction type (in-person or online)
    transaction_type = st.radio("Transaction Type", ["In-Person", "Online"])
    
    # Personal information
    col_a, col_b = st.columns(2)
    with col_a:
        gender = st.selectbox("Gender", ['M', 'F'])
    with col_b:
        birth_year = st.number_input("Birth Year", min_value=1930, max_value=2005, value=1980)
        birth_month = st.number_input("Birth Month", min_value=1, max_value=12, value=1)
        birth_day = st.number_input("Birth Day", min_value=1, max_value=31, value=1)
    
    # Location information with better defaults
    if transaction_type == "In-Person":
        st.subheader("Customer Location")
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=40.7)
        long = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-74.0)
        
        st.subheader("Merchant Location")
        merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=40.7)
        merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-74.1)
    else:  # Online transaction
        # Online-specific fields
        st.subheader("Online Transaction Details")
        is_new_device = st.checkbox("New Device", value=False)
        account_age_days = st.number_input("Account Age (days)", min_value=1, max_value=3650, value=180)
        is_digital_goods = st.checkbox("Digital Goods", value=False)
        billing_country = st.selectbox("Billing Country", ["US", "CA", "GB", "FR", "DE", "JP", "AU"])
        ip_country = st.selectbox("IP Country", ["US", "CA", "GB", "FR", "DE", "JP", "AU", "RU", "CN", "BR"])
        transaction_velocity = st.slider("Transactions in last 24h", 0, 20, 2)
        email_domain = st.text_input("Email Domain", "gmail.com")
        
        # Set location to None for online transactions
        lat = long = merch_lat = merch_long = None
    
    # Optional: Job (with full dropdown, no search)
    if 'job' in df.columns:
        jobs = sorted(df['job'].unique().tolist())
        job = st.selectbox("Job", jobs)
    else:
        job = None

with col2:
    st.subheader("Prediction Results")
    
    # This function replaces the existing predict_fraud() function with Bayesian integration
    def predict_fraud():
        current_use_saved_models = use_saved_models  # Create a local copy
        
        # Create transaction datetime string
        trans_datetime = pd.Timestamp(
            year=transaction_date.year,
            month=transaction_date.month,
            day=transaction_date.day,
            hour=transaction_time.hour,
            minute=transaction_time.minute
        )
        
        # Create input data
        input_data = {
            'trans_date_trans_time': [trans_datetime],
            'merchant': [merchant],
            'category': [category],
            'amt': [amount],
            'gender': [gender],
            'is_fraud': [0]  # Dummy value
        }
        
        # Add DOB if available
        if birth_year and birth_month and birth_day:
            # Create proper date object to avoid format issues
            dob = pd.Timestamp(year=birth_year, month=birth_month, day=birth_day)
            input_data['dob'] = [dob]
        
        # Add location data if available
        if lat is not None and long is not None:
            input_data['lat'] = [lat]
            input_data['long'] = [long]
            input_data['merch_lat'] = [merch_lat]
            input_data['merch_long'] = [merch_long]
        
        # Add job if available
        if job:
            input_data['job'] = [job]
        
        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        
        # ---------- BAYESIAN RISK CALCULATION BEGIN ----------
        
        # Initialize Bayesian calculator with appropriate prior
        # Online transactions typically have higher fraud rates
        prior_fraud_prob = 0.025 if transaction_type == "Online" else 0.01
        bayesian_calc = BayesianRiskCalculator(prior_fraud_prob=prior_fraud_prob)
        
        # Prepare data for Bayesian analysis
        if transaction_type == "In-Person":
            bayesian_data = {
                'hour': transaction_time.hour,
                'amount': amount,
                'category': category.lower(),
                'lat': lat,
                'long': long,
                'merch_lat': merch_lat,
                'merch_long': merch_long,
                'birth_year': birth_year,
                'gender': gender,
                'merchant': merchant
            }
            bayesian_result = bayesian_calc.calculate_inperson_risk(bayesian_data)
        else:  # Online transaction
            bayesian_data = {
                'hour': transaction_time.hour,
                'amount': amount,
                'category': category.lower(),
                'birth_year': birth_year,
                'is_new_device': is_new_device,
                'account_age_days': account_age_days,
                'is_digital_goods': is_digital_goods,
                'shipping_address': 'Digital delivery' if is_digital_goods else 'Shipping address',
                'billing_address': 'Billing address',
                'ip_country': ip_country,
                'billing_country': billing_country,
                'transaction_velocity': transaction_velocity,
                'email_domain': email_domain
            }
            bayesian_result = bayesian_calc.calculate_online_risk(bayesian_data)
        
        # Get the Bayesian probability
        bayesian_probability = bayesian_result['posterior_probability']
        
        # ---------- BAYESIAN RISK CALCULATION END ----------
        
        # Process input data - this will handle NaN values
        processed_input = preprocess_data(input_df)
        
        model = None
        model_probability = 0.0
        
        if current_use_saved_models and os.path.exists(os.path.join(models_path, model_choice.lower().replace(" ", "_") + ".pkl")):
            # Load saved model
            model_filename = model_choice.lower().replace(" ", "_") + ".pkl"
            model_path = os.path.join(models_path, model_filename)
            try:
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                    
                # Get the right columns from the model
                if hasattr(model, 'feature_names_in_'):
                    model_columns = model.feature_names_in_
                else:
                    # For older scikit-learn versions
                    # Sample a subset of data for getting columns
                    processed_df = preprocess_data(df)
                    X = processed_df.drop('is_fraud', axis=1)
                    model_columns = X.columns
                
                # Ensure input has same columns as the model requires
                for col in model_columns:
                    if col not in processed_input.columns:
                        processed_input[col] = 0
                
                # Keep only the columns needed for prediction and in the right order
                processed_input = processed_input.reindex(columns=model_columns, fill_value=0)
                
                # Check for NaN values and impute if necessary
                if processed_input.isna().any().any():
                    st.warning("Input data contains missing values. Applying imputation.")
                    num_cols = processed_input.select_dtypes(include=['float64', 'int64']).columns
                    if len(num_cols) > 0:
                        num_imputer = SimpleImputer(strategy='median')
                        processed_input[num_cols] = num_imputer.fit_transform(processed_input[num_cols])
                
                # Make prediction and get probability
                prediction = model.predict(processed_input)
                model_probability = model.predict_proba(processed_input)[0][1]
            except Exception as e:
                st.error(f"Error using saved model: {e}")
                # Fallback to training a new model
                current_use_saved_models = False
                
        if not current_use_saved_models or not os.path.exists(os.path.join(models_path, model_choice.lower().replace(" ", "_") + ".pkl")):
            # Process the dataset - removed sampling to use full dataset
            processed_df = preprocess_data(df)
            
            # Train-test split
            X = processed_df.drop('is_fraud', axis=1)
            y = processed_df['is_fraud']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Apply SMOTE if selected - with better handling of large datasets
            use_smote_actual = False
            if use_smote:
                try:
                    if len(X_train) > 100000:  # Arbitrary threshold for "large" dataset
                        st.warning("Dataset too large for SMOTE. Training without SMOTE.")
                    else:
                        # Check for NaN values and impute if necessary
                        if X_train.isna().any().any():
                            st.warning("Training data contains missing values. Applying imputation before SMOTE.")
                            num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
                            num_imputer = SimpleImputer(strategy='median')
                            X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
                        
                        smote = SMOTE(sampling_strategy=0.5, k_neighbors=min(5, sum(y_train)-1), random_state=42)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        use_smote_actual = True
                except Exception as e:
                    st.warning(f"SMOTE could not be applied: {str(e)}")
    
            # Model training based on user choice
            if model_choice == 'Logistic Regression':
                model = LogisticRegression(max_iter=1000, class_weight=None if use_smote_actual else 'balanced')
            elif model_choice == 'Random Forest':
                model = RandomForestClassifier(n_estimators=100, class_weight=None if use_smote_actual else 'balanced')
            else:  # XGBoost
                if use_smote_actual:
                    model = XGBClassifier(eval_metric='logloss')
                else:
                    model = XGBClassifier(eval_metric='logloss', 
                                        scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1])
    
            model.fit(X_train, y_train)
            
            # Ensure input has same columns as training data
            for col in X.columns:
                if col not in processed_input.columns:
                    processed_input[col] = 0
            
            # Keep only the columns needed for prediction and in the right order
            processed_input = processed_input.reindex(columns=X.columns, fill_value=0)
            
            # Check for NaN values in the input data
            if processed_input.isna().any().any():
                st.warning("Input data contains missing values. Applying imputation.")
                num_cols = processed_input.select_dtypes(include=['float64', 'int64']).columns
                if len(num_cols) > 0:
                    num_imputer = SimpleImputer(strategy='median')
                    processed_input[num_cols] = num_imputer.fit_transform(processed_input[num_cols])
            
            # Make prediction and get probability
            prediction = model.predict(processed_input)
            model_probability = model.predict_proba(processed_input)[0][1]
        
        # ----------------- HYBRID RISK SCORE -----------------
        # Combine model prediction with Bayesian probability
        # Weight can be adjusted based on confidence in each approach
        model_weight = 1.0 - bayesian_weight
        bayesian_weight_final = bayesian_weight
        hybrid_probability = (model_weight * model_probability) + (bayesian_weight_final * bayesian_probability)
        
        # Apply threshold to get final prediction
        final_prediction = 1 if hybrid_probability >= risk_threshold else 0
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = processed_input.columns
            feature_importance = dict(zip(features, importances))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0]
            features = processed_input.columns
            feature_importance = dict(zip(features, np.abs(coef)))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_features = None
        
        # Create a summary of risk factors with their likelihood ratios
        risk_factors = bayesian_result['individual_likelihoods']
        
        # Convert risk factors to a more readable format
        readable_risk_factors = {}
        for factor, value in risk_factors.items():
            readable_name = factor.replace('_', ' ').title()
            readable_risk_factors[readable_name] = value
        
        # Create a hash of the transaction data for blockchain
        transaction_hash = hashlib.sha256(
            f"{merchant}|{category}|{amount}|{trans_datetime}|{lat}|{long}".encode()
        ).hexdigest()
        
        # Create detailed risk information for display
        risk_details = {
            "model_probability": model_probability,
            "bayesian_probability": bayesian_probability,
            "hybrid_probability": hybrid_probability,
            "prior_probability": bayesian_result['prior_probability'],
            "likelihood_ratio": bayesian_result['likelihood_ratio'],
            "individual_likelihoods": risk_factors,
            "transaction_type": transaction_type
        }
        
        # Record prediction on blockchain
        blockchain_tx_hash = None
        if web3 and contract and account:
            try:
                # Convert probability to a percentage (0-100)
                confidence = int(hybrid_probability * 100)
                
                # Call the contract method
                tx_hash = contract.functions.recordPrediction(
                    bool(final_prediction),
                    confidence,
                    transaction_hash
                ).transact({'from': account, 'gas': 500000})
                
                # Wait for transaction receipt
                receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Get the transaction hash
                blockchain_tx_hash = receipt['transactionHash'].hex()
            except Exception as e:
                st.error(f"Blockchain transaction failed: {e}")
        
        return final_prediction, hybrid_probability, top_features, readable_risk_factors, blockchain_tx_hash, risk_details

    # Determine button text based on blockchain availability
    button_text = "Predict Fraud and Record on Blockchain" if web3 and contract and account else "Predict Fraud"
    button_type = "primary"
    
    # Predict button
    if st.button(button_text, type=button_type):
        # Check blockchain connection when attempting to use it
        blockchain_warning = False
        if button_text == "Predict Fraud and Record on Blockchain":
            if not web3 or not web3.is_connected():
                st.warning("Blockchain connection not available. Proceeding with prediction only.")
                blockchain_warning = True
            elif not account:
                st.warning("No account selected. Proceeding with prediction only.")
                blockchain_warning = True
            elif not contract:
                st.warning("Contract not properly loaded. Proceeding with prediction only.")
                blockchain_warning = True
        
        with st.spinner("Analyzing transaction..."):
            # Add a progress bar to simulate processing
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            try:
                prediction_results = predict_fraud()
                
                # Unpack results (handle the expanded return values with risk_details)
                if len(prediction_results) == 6:
                    prediction, probability, top_features, risk_factors, tx_hash, risk_details = prediction_results
                else:
                    prediction, probability, top_features, risk_factors = prediction_results[:4]
                    tx_hash = None
                    risk_details = None
                
                # Display results with better visualization
                if prediction == 1:
                    st.error("⚠️ **FRAUDULENT TRANSACTION DETECTED!**")
                    st.markdown(f"<h3 style='color:red'>Risk Score: {probability*100:.2f}%</h3>", unsafe_allow_html=True)
                else:
                    st.success("✅ **LEGITIMATE TRANSACTION**")
                    st.markdown(f"<h3 style='color:green'>Risk Score: {probability*100:.2f}%</h3>", unsafe_allow_html=True)
                
                # Display blockchain transaction information if available
                if tx_hash:
                    st.success("Prediction recorded on blockchain")
                    st.info(f"Transaction Hash: {tx_hash}")
                    st.markdown(f"View on local blockchain explorer (if available)")
                elif button_text == "Predict Fraud and Record on Blockchain" and not blockchain_warning:
                    st.warning("Prediction was not recorded on blockchain")
                
                # Create tabs for detailed information - add Bayesian tab
                tab1, tab2, tab3, tab4 = st.tabs(["Transaction Summary", "Risk Analysis", "Bayesian Details", "Model Details"])
                
                with tab1:
                    # Display transaction details in a more visually appealing way
                    st.subheader("Transaction Details")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Merchant:** {merchant}")
                        st.markdown(f"**Category:** {category}")
                        st.markdown(f"**Amount:** ${amount:.2f}")
                        st.markdown(f"**Transaction Type:** {transaction_type}")
                    
                    with col2:
                        st.markdown(f"**Date & Time:** {transaction_date.strftime('%Y-%m-%d')} {transaction_time.strftime('%H:%M')}")
                        st.markdown(f"**Gender:** {gender}")
                        st.markdown(f"**Birth Date:** {birth_year}-{birth_month:02d}-{birth_day:02d}")
                
                with tab2:
                    # Display risk factors
                    st.subheader("Risk Factor Analysis")
                    
                    # Create a bar chart for risk factors
                    risk_data = pd.DataFrame({
                        'Factor': list(risk_factors.keys()),
                        'Risk Multiplier': list(risk_factors.values())
                    })
                    
                    # Sort by risk multiplier to show highest risks first
                    risk_data = risk_data.sort_values('Risk Multiplier', ascending=False)
                    
                    st.bar_chart(risk_data.set_index('Factor'))
                    
                    # Highlight the highest risk factors
                    st.subheader("Key Risk Indicators")
                    for factor, value in risk_factors.items():
                        if value > 1.0:
                            st.warning(f"⚠️ **{factor}**: Risk multiplier {value:.1f}x")
                        else:
                            st.info(f"✓ **{factor}**: Normal risk level")
                    
                    # Display top model features if available
                    if top_features:
                        st.subheader("Top Model Features")
                        for feature, importance in top_features:
                            st.write(f"- {feature}: {importance:.4f}")
                
                with tab3:
                    # New tab for Bayesian analysis details
                    st.subheader("Bayesian Risk Analysis")
                    
                    if risk_details:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Probability Breakdown")
                            st.markdown(f"**Prior Probability:** {risk_details['prior_probability']:.2%}")
                            st.markdown(f"**ML Model Probability:** {risk_details['model_probability']:.2%}")
                            st.markdown(f"**Bayesian Probability:** {risk_details['bayesian_probability']:.2%}")
                            st.markdown(f"**Final Hybrid Probability:** {risk_details['hybrid_probability']:.2%}")
                            
                            # Create a small chart to compare probabilities
                            probs = {
                                'Prior': risk_details['prior_probability'],
                                'ML Model': risk_details['model_probability'],
                                'Bayesian': risk_details['bayesian_probability'],
                                'Hybrid': risk_details['hybrid_probability']
                            }
                            
                            probs_df = pd.DataFrame({
                                'Probability Type': list(probs.keys()),
                                'Value': list(probs.values())
                            })
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            bars = ax.bar(probs_df['Probability Type'], probs_df['Value'], color=['#90CAF9', '#A5D6A7', '#FFAB91', '#CE93D8'])
                            
                            # Add value labels
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                        f'{height:.2%}', ha='center', va='bottom')
                            
                            ax.set_ylim(0, max(probs.values()) * 1.2)
                            ax.set_title('Probability Comparison')
                            ax.set_ylabel('Probability')
                            ax.grid(axis='y', linestyle='--', alpha=0.7)
                            
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("### Likelihood Ratio Analysis")
                            st.markdown(f"**Combined Likelihood Ratio:** {risk_details['likelihood_ratio']:.2f}x")
                            
                            # Create data for likelihood ratio chart
                            likelihoods = risk_details['individual_likelihoods']
                            likelihood_data = pd.DataFrame({
                                'Factor': list(likelihoods.keys()),
                                'Likelihood Ratio': list(likelihoods.values())
                            })
                            
                            # Sort by likelihood ratio to show highest risks first
                            likelihood_data = likelihood_data.sort_values('Likelihood Ratio', ascending=False)
                            
                            # Plot likelihood ratios as a horizontal bar chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            bars = ax.barh(likelihood_data['Factor'], likelihood_data['Likelihood Ratio'], color='skyblue')
                            
                            # Add value labels on the bars
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.1f}x', 
                                       va='center', fontsize=8)
                            
                            # Add a vertical line at x=1 (baseline)
                            ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Baseline Risk')
                            
                            ax.set_title('Risk Likelihood Ratios by Factor')
                            ax.set_xlabel('Likelihood Ratio (higher = more risky)')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                        
                        # Bayesian interpretation
                        st.subheader("Bayesian Interpretation")
                        st.markdown("""
                        **How to interpret Bayesian risk scores:**
                        
                        - **Prior Probability** is the base rate of fraud before considering any evidence
                        - **Likelihood Ratios** show how much more likely each risk factor is in fraudulent vs. legitimate transactions
                        - **Bayesian Probability** is the updated fraud probability after considering all evidence
                        - **Hybrid Probability** combines the ML model prediction with the Bayesian analysis
                        
                        The final risk score uses both statistical patterns from your ML model and transaction-specific 
                        risk factors to give a more comprehensive fraud assessment.
                        """)
                    else:
                        st.warning("Bayesian risk details not available")
                
                with tab4:
                    # Model details
                    st.subheader("Model Configuration")
                    st.write(f"**Model:** {model_choice}")
                    st.write(f"**SMOTE Applied:** {'Yes' if use_smote else 'No'}")
                    st.write(f"**Risk Threshold:** {risk_threshold}")
                    st.write(f"**Bayesian Weight:** {bayesian_weight}")
                    
                    # Provide context based on fraud patterns
                    st.subheader("Detection Explanation")
                    if prediction == 1:
                        st.write("""
                        This transaction was flagged as potentially fraudulent based on:
                        
                        1. The machine learning model's prediction based on statistical patterns
                        2. Bayesian analysis of specific risk factors
                        3. Comparison with known fraud patterns
                        
                        **Recommended Action:** Verify this transaction with the cardholder before approval.
                        """)
                    else:
                        st.write("""
                        This transaction appears legitimate based on:
                        
                        1. Consistency with normal transaction patterns
                        2. Low risk factors across key dimensions
                        3. High confidence score from both model and Bayesian analysis
                        
                        **Recommended Action:** Process transaction normally.
                        """)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please check your input data and try again.")

# Display Blockchain information
if web3 and web3.is_connected() and contract:
    st.sidebar.title("Blockchain Status")
    try:
        prediction_count = contract.functions.getPredictionCount().call()
        st.sidebar.write(f"Total Predictions Recorded: {prediction_count}")
        
        if prediction_count > 0:
            st.sidebar.subheader("Recent Predictions")
            for i in range(min(5, prediction_count)):
                prediction_id = contract.functions.getPredictionIdAtIndex(prediction_count - i - 1).call()
                prediction_data = contract.functions.getPrediction(prediction_id).call()
                
                is_fraud = "⚠️ Fraud" if prediction_data[1] else "✅ Legitimate"
                confidence = prediction_data[2]
                
                st.sidebar.write(f"{i+1}. {is_fraud} - {confidence}% confidence")
    except Exception as e:
        st.sidebar.error(f"Error retrieving blockchain data: {e}")