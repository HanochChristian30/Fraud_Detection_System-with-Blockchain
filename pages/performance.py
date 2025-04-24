import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
# Add SMOTE import
from imblearn.over_sampling import SMOTE

# Set page configuration
st.set_page_config(page_title="Fraud Detection - Performance", layout="wide")

# Page title
st.title("Model Performance Evaluation")
st.write("Compare performance metrics across different fraud detection models.")

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
        processed_df['trans_date_trans_time'] = pd.to_datetime(processed_df['trans_date_trans_time'])
        processed_df['hour'] = processed_df['trans_date_trans_time'].dt.hour
        processed_df['day'] = processed_df['trans_date_trans_time'].dt.day
        processed_df['month'] = processed_df['trans_date_trans_time'].dt.month
        processed_df['dayofweek'] = processed_df['trans_date_trans_time'].dt.dayofweek
        processed_df.drop(['trans_date_trans_time'], axis=1, inplace=True)
    
    # Convert gender to numeric
    if 'gender' in processed_df.columns:
        processed_df['gender'] = processed_df['gender'].map({'M': 1, 'F': 0})
    
    # Handle date of birth
    if 'dob' in processed_df.columns:
        processed_df['dob'] = pd.to_datetime(processed_df['dob'])
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
    
    return processed_df

# Load data
df = load_data()

# Let user inspect the data
with st.expander("Preview Dataset"):
    st.write(df.head())
    
    # Provide data information
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Shape:", df.shape)
    with col2:
        if 'is_fraud' in df.columns:
            fraud_count = df['is_fraud'].sum()
            st.write(f"Fraud Transactions: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")

# Check if required columns exist
required_columns = ['is_fraud']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

# Sample size control
sample_size = st.slider("Sample Size (% of data)", 1, 100, 10, help="Using a smaller sample speeds up processing")
if sample_size < 100:
    df = df.sample(frac=sample_size/100, random_state=42)
    st.info(f"Using {len(df)} transactions ({sample_size}% of data)")

try:
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        processed_df = preprocess_data(df)
        
        # Display feature importance calculation option
        calculate_importance = st.checkbox("Calculate Feature Importance", value=False)
        
        # Display processed features
        with st.expander("View processed features"):
            st.write("Features used for modeling:", processed_df.columns.tolist())
            st.write("Processed data preview:", processed_df.head())
    
    # Train-test split
    X = processed_df.drop('is_fraud', axis=1)
    y = processed_df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Display class distribution before SMOTE
    st.subheader("Class Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before SMOTE:**")
        st.write(f"Training set: {y_train.value_counts()[0]} legitimate, {y_train.value_counts()[1]} fraudulent transactions")
        st.write(f"Testing set: {y_test.value_counts()[0]} legitimate, {y_test.value_counts()[1]} fraudulent transactions")
    
    # Add SMOTE options
    use_smote = st.checkbox("Apply SMOTE to address class imbalance", value=True)
    
    if use_smote:
        with st.spinner("Applying SMOTE to balance classes..."):
            # SMOTE settings in sidebar
            sampling_strategy = st.select_slider(
                "SMOTE Sampling Strategy",
                options=[0.1, 0.25, 0.5, 0.75, 1.0],
                value=0.5,
                help="Ratio of minority to majority class after resampling (1.0 = fully balanced)"
            )
            
            k_neighbors = st.slider("SMOTE k_neighbors", 1, 10, 5, 
                                    help="Number of nearest neighbors to use for synthetic samples")
            
            # Apply SMOTE to training data only
            smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
            X_train_resampled, y_train_resampled = smote, y_train
            
            try:
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                with col2:
                    st.write("**After SMOTE:**")
                    st.write(f"Training set: {np.sum(y_train_resampled == 0)} legitimate, {np.sum(y_train_resampled == 1)} fraudulent transactions")
                    st.write(f"Testing set: {y_test.value_counts()[0]} legitimate, {y_test.value_counts()[1]} fraudulent transactions (unchanged)")
            except Exception as e:
                st.error(f"SMOTE error: {str(e)}")
                st.warning("Falling back to original imbalanced dataset")
                X_train_resampled, y_train_resampled = X_train, y_train
    else:
        # Use original data if SMOTE is not selected
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced' if not use_smote else None),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced' if not use_smote else None),
        "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1] if not use_smote else 1)
    }
    
    # Model selection
    selected_models = st.multiselect(
        "Select Models to Evaluate",
        list(models.keys()),
        default=list(models.keys())
    )
    
    if not selected_models:
        st.warning("Please select at least one model to evaluate.")
    else:
        with st.spinner("Training and evaluating models..."):
            # Train models and evaluate performance metrics
            performance_metrics = {}
            model_predictions = {}
            model_probabilities = {}
            trained_models = {}
            
            for name in selected_models:
                model = models[name]
                model.fit(X_train_resampled, y_train_resampled)  # Use resampled data
                trained_models[name] = model
                
                y_pred = model.predict(X_test)
                model_predictions[name] = y_pred
                
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    model_probabilities[name] = y_prob
                except:
                    model_probabilities[name] = None
                
                performance_metrics[name] = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, zero_division=0)
                }
            
            # Display metrics in a comparison table
            st.subheader("Performance Metrics Comparison")
            metrics_df = pd.DataFrame(performance_metrics)
            st.dataframe(metrics_df.style.highlight_max(axis=1), use_container_width=True)
            
            # Feature importance
            if calculate_importance and selected_models:
                st.subheader("Feature Importance")
                
                for name in selected_models:
                    model = trained_models[name]
                    
                    # Get feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        
                        # Get feature names
                        feature_names = X.columns
                        
                        # Show top 15 features
                        top_n = min(15, len(feature_names))
                        
                        st.write(f"Top {top_n} features for {name}:")
                        fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
                        ax.barh(range(top_n), importances[indices][:top_n], align='center')
                        ax.set_yticks(range(top_n))
                        ax.set_yticklabels([feature_names[i] for i in indices][:top_n])
                        ax.set_xlabel('Relative Importance')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    elif name == "Logistic Regression":
                        # For logistic regression, use coefficients
                        coef = model.coef_[0]
                        indices = np.argsort(np.abs(coef))[::-1]
                        
                        # Get feature names
                        feature_names = X.columns
                        
                        # Show top 15 features
                        top_n = min(15, len(feature_names))
                        
                        st.write(f"Top {top_n} features for {name}:")
                        fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
                        ax.barh(range(top_n), np.abs(coef[indices][:top_n]), align='center')
                        ax.set_yticks(range(top_n))
                        ax.set_yticklabels([feature_names[i] for i in indices][:top_n])
                        ax.set_xlabel('Coefficient Magnitude')
                        plt.tight_layout()
                        st.pyplot(fig)
            
            # Visualization tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics Chart", "🔍 Confusion Matrices", "📈 ROC Curves", "🧪 SMOTE Analysis"])
            
            with tab1:
                # Bar chart for metrics comparison
                for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                    st.subheader(f"{metric} Comparison")
                    fig, ax = plt.figure(figsize=(10, 5)), plt.axes()
                    ax.bar(selected_models, [performance_metrics[model][metric] for model in selected_models])
                    ax.set_ylabel(metric)
                    ax.set_ylim(0, 1)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            with tab2:
                # Confusion matrices
                for name in selected_models:
                    st.subheader(f"Confusion Matrix: {name}")
                    cm = confusion_matrix(y_test, model_predictions[name])
                    
                    fig, ax = plt.figure(figsize=(6, 5)), plt.axes()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=['Legitimate', 'Fraud'],
                                yticklabels=['Legitimate', 'Fraud'])
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    st.pyplot(fig)
                    
                    # Interpretation
                    tn, fp, fn, tp = cm.ravel()
                    st.text(f"""
                    True Negatives (correctly identified legitimate): {tn}
                    False Positives (incorrectly flagged as fraud): {fp}
                    False Negatives (missed fraud): {fn}
                    True Positives (correctly identified fraud): {tp}
                    """)
                    
                    # Business impact analysis
                    st.subheader("Business Impact Analysis")
                    
                    # Assume average transaction amount and cost values
                    avg_transaction = 100  # Example value
                    false_positive_cost = 5  # Cost of investigating a false alert
                    false_negative_cost = 75  # Cost of undetected fraud (% of transaction)
                    
                    # Calculate costs
                    investigation_cost = fp * false_positive_cost
                    fraud_loss = fn * avg_transaction * (false_negative_cost/100)
                    total_cost = investigation_cost + fraud_loss
                    
                    st.write(f"""
                    **Estimated Costs:**
                    - False Positive Cost (unnecessary investigations): ${investigation_cost:.2f}
                    - False Negative Cost (undetected fraud): ${fraud_loss:.2f}
                    - Total Operational Cost: ${total_cost:.2f}
                    """)
            
            with tab3:
                # ROC curves
                st.subheader("ROC Curves")
                fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
                
                for name in selected_models:
                    if model_probabilities[name] is not None:
                        fpr, tpr, _ = roc_curve(y_test, model_probabilities[name])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
                
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                ax.grid(linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                
                st.info("""
                **ROC Curve Interpretation:** 
                The closer the curve follows the top-left corner, the better the model's performance. 
                A higher AUC value indicates a better model at distinguishing between fraud and legitimate transactions.
                """)
                
            with tab4:
                # SMOTE Analysis
                st.subheader("SMOTE Impact Analysis")
                
                if use_smote:
                    st.write("""
                    ### How SMOTE Addresses Overfitting
                    
                    **Class Imbalance Problem:**
                    - In fraud detection, fraudulent transactions (minority class) are typically much rarer than legitimate ones
                    - Models tend to become biased toward the majority class, leading to high accuracy but poor fraud detection
                    
                    **SMOTE Benefits:**
                    - Creates synthetic samples of the minority class rather than simple oversampling
                    - Improves model's ability to identify fraud patterns without memorizing specific cases
                    - Helps balance precision and recall for better overall performance
                    
                    **Current Settings:**
                    - Sampling Strategy: {sampling_strategy} (this controls how balanced the classes become)
                    - K-Neighbors: {k_neighbors} (number of neighbors used to generate synthetic samples)
                    """)
                    
                    # Show class distribution visually
                    fig, ax = plt.figure(figsize=(10, 5)), plt.axes()
                    
                    # Original distribution
                    original_counts = [y_train.value_counts()[0], y_train.value_counts()[1]]
                    
                    # SMOTE distribution
                    smote_counts = [np.sum(y_train_resampled == 0), np.sum(y_train_resampled == 1)]
                    
                    ax.bar(['Original Legitimate', 'Original Fraud', 'SMOTE Legitimate', 'SMOTE Fraud'],
                           original_counts + smote_counts,
                           color=['blue', 'blue', 'orange', 'orange'])
                    
                    ax.set_ylabel('Number of Transactions')
                    ax.set_title('Class Distribution: Before vs After SMOTE')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    st.write("""
                    ### Tuning Recommendations
                    
                    **For Higher Precision (fewer false positives):**
                    - Decrease the sampling strategy (e.g., 0.25 instead of 0.5)
                    - Use higher k_neighbors value to create more conservative synthetic samples
                    
                    **For Higher Recall (catch more fraud):**
                    - Increase sampling strategy closer to 1.0 for more balanced classes
                    - Consider combining SMOTE with model threshold adjustments
                    """)
                else:
                    st.write("""
                    SMOTE is currently disabled. Enable it above to see how synthetic oversampling 
                    can help address class imbalance and reduce overfitting.
                    """)

except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    st.write("Please check your dataset structure and try again.")