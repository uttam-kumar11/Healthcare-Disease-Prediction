import streamlit as st
import pandas as pd
import numpy as np
import joblib # For loading the saved models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import hashlib # For password hashing
import uuid # For generating anonymous user IDs
from datetime import datetime # For timestamping predictions
import json # Added for json.loads

# --- Firebase Imports and Setup (for Firestore) ---
# These global variables are provided in the Canvas environment, but not in local Python.
# For local testing, we'll assign default values.
try:
    app_id = __app_id
except NameError:
    app_id = 'default-app-id' # Placeholder for local testing

try:
    # Use json.loads for Python to parse JSON strings
    # Ensure __firebase_config is defined and is a string before attempting json.loads
    if '__firebase_config' in locals() or '__firebase_config' in globals():
        if isinstance(__firebase_config, str):
            firebase_config = json.loads(__firebase_config) # Fixed: use firebase_config not __firebase_config
        else:
            firebase_config = __firebase_config # If it's already parsed or not a string
    else:
        firebase_config = None # Placeholder for local testing if not defined
except (json.JSONDecodeError, TypeError): # Catch JSONDecodeError if not valid JSON, TypeError if not string
    firebase_config = None # Handle if __firebase_config is not a valid JSON string or type

try:
    initial_auth_token = __initial_auth_token
except NameError:
    initial_auth_token = None # Placeholder for local testing

# Placeholder for a "database" (in a real app, this would be Firestore)
# We'll store predictions in session state for demonstration of "past predictions"
if 'predictions_db' not in st.session_state:
    st.session_state['predictions_db'] = []

# --- 1. Load Trained Models ---
try:
    # Diabetes Models - Loading OPTIMIZED versions
    log_reg_model = joblib.load('log_reg_model_optimized.pkl')
    rf_model = joblib.load('rf_model_optimized.pkl')
    dt_model = joblib.load('dt_model_optimized.pkl')

    # Heart Disease Models - Loading OPTIMIZED versions
    heart_log_reg_model = joblib.load('heart_log_reg_model_optimized.pkl')
    heart_rf_model = joblib.load('heart_rf_model_optimized.pkl')
    heart_dt_model = joblib.load('heart_dt_model_optimized.pkl')

    st.success("All Machine Learning Models (Optimized) loaded successfully!")
except FileNotFoundError:
    st.error("One or more OPTIMIZED model files not found. Please ensure all _optimized.pkl files are in the same directory as app.py.")
    st.stop()

models_diabetes = {
    "Logistic Regression": log_reg_model,
    "Random Forest": rf_model,
    "Decision Tree": dt_model
}

models_heart_disease = {
    "Logistic Regression": heart_log_reg_model,
    "Random Forest": heart_rf_model,
    "Decision Tree": heart_dt_model
}

# --- User Authentication Setup ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

user_db = {
    "user1": make_hashes("pass123"),
    "doctor": make_hashes("health@123"),
    "admin": make_hashes("admin@123")
}

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

def login_page():
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        hashed_password = make_hashes(password)
        if username in user_db and user_db[username] == hashed_password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['user_id'] = f"auth_{username}"
            st.sidebar.success(f"Logged in as {username}")
            st.rerun()
        else:
            st.sidebar.error("Invalid Username or Password")

def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['user_id'] = str(uuid.uuid4())
        st.session_state['predictions_db'] = []
        st.sidebar.info("Logged out successfully.")
        st.rerun()

# --- 2. Define Preprocessing Functions (consistent with notebook) ---
def preprocess_diabetes_input(input_df):
    cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_check:
        input_df[col] = input_df[col].replace(0, np.nan)
        if col == 'Glucose': median_val = 117.0
        elif col == 'BloodPressure': median_val = 72.0
        elif col == 'SkinThickness': median_val = 29.0
        elif col == 'Insulin': median_val = 125.0
        elif col == 'BMI': median_val = 32.3
        else: median_val = input_df[col].median()
        input_df[col] = input_df[col].fillna(median_val)
    return input_df

def preprocess_heart_input(input_df):
    for col in ['ca', 'thal']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(input_df[col].mode()[0]).astype(float)
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(float)
    temp_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=int)
    heart_feature_columns_order = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_1.0', 'cp_2.0', 'cp_3.0', 'cp_4.0', 'fbs_1.0', 'restecg_1.0', 'restecg_2.0', 'exang_1.0', 'slope_2.0', 'slope_3.0', 'ca_1.0', 'ca_2.0', 'ca_3.0', 'thal_6.0', 'thal_7.0'
    ]
    final_input_df = pd.DataFrame(columns=heart_feature_columns_order)
    for col in heart_feature_columns_order:
        if col in temp_df.columns:
            final_input_df[col] = temp_df[col]
        else:
            final_input_df[col] = 0
    return final_input_df

# --- Global Data Parameters (moved outside login block) ---
data_url_diabetes = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
cols_to_check_full_diabetes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
diabetes_medians = { 'Glucose': 117.0, 'BloodPressure': 72.0, 'SkinThickness': 29.0, 'Insulin': 125.0, 'BMI': 32.3 }
target_col_diabetes = 'Outcome'
target_labels_diabetes = ['No Diabetes', 'Diabetes']
feature_cols_order_diabetes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

data_url_heart = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_column_names_full = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]
heart_modes = { 'ca': 0.0, 'thal': 3.0 }
target_col_heart = 'num'
target_labels_heart = ['No Heart Disease', 'Heart Disease']
heart_feature_columns_order = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_1.0', 'cp_2.0', 'cp_3.0', 'cp_4.0', 'fbs_1.0', 'restecg_1.0', 'restecg_2.0', 'exang_1.0', 'slope_2.0', 'slope_3.0', 'ca_1.0', 'ca_2.0', 'ca_3.0', 'thal_6.0', 'thal_7.0'
]

# --- Streamlit App Layout ---
st.set_page_config(page_title="Disease Prediction App", layout="wide", initial_sidebar_state="expanded")

# --- App Header ---
st.title("🏥 Healthcare Analytics: Disease Prediction")
st.markdown("---")


tab1, tab2, tab3 = st.tabs(["Single Patient Prediction", "Batch Prediction (CSV)", "Past Predictions"])


# Main app logic based on login status
if not st.session_state['logged_in']:
    login_page()
else:
    st.sidebar.success(f"Welcome, {st.session_state['username']}!")
    logout_button()

    st.sidebar.header("App Configuration")

    disease_selection = st.sidebar.selectbox(
        "Select Disease for Prediction:",
        ["Diabetes", "Heart Disease"]
    )

    if disease_selection == "Diabetes":
        selected_models = models_diabetes
        current_data_url_full = data_url_diabetes
        current_cols_to_check_full = cols_to_check_full_diabetes
        current_medians_modes = diabetes_medians
        current_target_col = target_col_diabetes
        current_target_labels = target_labels_diabetes
        current_feature_cols_order = feature_cols_order_diabetes
        current_preprocess_func = preprocess_diabetes_input
    else: # Heart Disease
        selected_models = models_heart_disease
        current_data_url_full = data_url_heart
        current_cols_to_check_full = heart_column_names_full
        current_medians_modes = heart_modes
        current_target_col = target_col_heart
        current_target_labels = target_labels_heart
        current_feature_cols_order = heart_feature_columns_order
        current_preprocess_func = preprocess_heart_input


    selected_model_name = st.sidebar.selectbox(
        f"Select Machine Learning Model for {disease_selection}:",
        list(selected_models.keys())
    )
    selected_model = selected_models[selected_model_name]
    st.sidebar.info(f"Currently selected model: **{selected_model_name}**")

    st.sidebar.markdown("---")
    st.sidebar.header("About This App")
    st.sidebar.write("""
    This application uses machine learning to predict the likelihood of various diseases based on health parameters.
    It's built with Streamlit for an interactive and user-friendly experience.
    """)
    st.sidebar.markdown("---")


    with tab1:
        st.header(f"Predict {disease_selection} Risk for a Single Patient")

        st.subheader("Enter Patient Health Data:")
        
        user_input_data = {}
        if disease_selection == "Diabetes":
            col1, col2, col3 = st.columns(3)
            with col1:
                user_input_data['Pregnancies'] = st.number_input("Pregnancies", min_value=0, max_value=17, value=1, key="preg_single")
                user_input_data['Glucose'] = st.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=120, key="glu_single")
                user_input_data['BloodPressure'] = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=122, value=70, key="bp_single")
            with col2:
                user_input_data['SkinThickness'] = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20, key="st_single")
                user_input_data['Insulin'] = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=846, value=80, key="ins_single")
                user_input_data['BMI'] = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=67.1, value=25.0, format="%.1f", key="bmi_single")
            with col3:
                user_input_data['DiabetesPedigreeFunction'] = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.3, format="%.3f", key="dpf_single")
                user_input_data['Age'] = st.number_input("Age (years)", min_value=21, max_value=81, value=30, key="age_single")
            
            user_df = pd.DataFrame([user_input_data])
            processed_user_df = current_preprocess_func(user_df.copy())

        else: # Heart Disease
            col1, col2, col3 = st.columns(3)
            with col1:
                user_input_data['age'] = st.number_input("Age (years)", min_value=0, max_value=120, value=50, key="age_heart_single")
                user_input_data['sex'] = float(st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="sex_heart_single"))
                user_input_data['cp'] = float(st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], key="cp_heart_single"))
                user_input_data['trestbps'] = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=200, value=120, key="trestbps_heart_single")
                user_input_data['chol'] = st.number_input("Serum Cholestoral (chol)", min_value=0, max_value=600, value=200, key="chol_heart_single")
            with col2:
                user_input_data['fbs'] = float(st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True", key="fbs_heart_single") )
                user_input_data['restecg'] = float(st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2], key="restecg_heart_single"))
                user_input_data['thalach'] = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=220, value=150, key="thalach_heart_single")
                user_input_data['exang'] = float(st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="exang_heart_single"))
                user_input_data['oldpeak'] = st.number_input("ST depression induced by exercise relative to rest (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, format="%.1f", key="oldpeak_heart_single")
            with col3:
                user_input_data['slope'] = float(st.selectbox("Slope of the peak exercise ST segment (slope)", options=[0, 1, 2], key="slope_heart_single"))
                user_input_data['ca'] = float(st.selectbox("Number of major vessels (ca)", options=[0, 1, 2, 3], key="ca_heart_single"))
                user_input_data['thal'] = float(st.selectbox("Thalium Stress Test Result (thal)", options=[1, 2, 3], key="thal_heart_single")) # Note: 0 is not a valid option here typically after preprocessing
            
            user_df = pd.DataFrame([user_input_data])
            processed_user_df = current_preprocess_func(user_df.copy())
            
            # Apply scaling if Logistic Regression is selected for Heart Disease
            if selected_model_name == "Logistic Regression":
                # Re-load full heart disease data for scaler fitting (for consistency)
                # This block is problematic as it re-loads and re-processes the entire dataset on every prediction
                # It also fits the scaler on the full dataset, not just training.
                # A better approach is to save/load the fitted scaler from the notebook.
                df_heart_full_scaler = pd.read_csv(current_data_url_full, names=heart_column_names_full, na_values='?')
                for col in ['ca', 'thal']:
                    df_heart_full_scaler[col] = pd.to_numeric(df_heart_full_scaler[col], errors='coerce')
                    mode_val = df_heart_full_scaler[col].mode()[0]
                    df_heart_full_scaler[col] = df_heart_full_scaler[col].fillna(mode_val)
                df_heart_full_scaler['num'] = df_heart_full_scaler['num'].apply(lambda x: 1 if x > 0 else 0)
                categorical_cols_full = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
                # MODIFICATION: Ensure full dataset categorical columns are float type before get_dummies
                for col in categorical_cols_full:
                    if col in df_heart_full_scaler.columns:
                        df_heart_full_scaler[col] = df_heart_full_scaler[col].astype(float)
                df_heart_full_scaler = pd.get_dummies(df_heart_full_scaler, columns=categorical_cols_full, drop_first=True, dtype=int)
                
                X_heart_full_scaler = df_heart_full_scaler.drop('num', axis=1)
                
                # Ensure X_heart_full_scaler columns match the training order
                X_heart_full_scaler = X_heart_full_scaler.reindex(columns=heart_feature_columns_order, fill_value=0)

                scaler_heart = StandardScaler()
                scaler_heart.fit(X_heart_full_scaler) # Fit scaler on the full dataset
                processed_user_df = pd.DataFrame(scaler_heart.transform(processed_user_df), columns=processed_user_df.columns)


        # Add a predict button
        predict_button = st.button("Predict Risk", key="predict_single_disease")

        # --- Prediction Logic ---
        if predict_button:
            st.subheader("Prediction Results:")
            
            # Make prediction
            prediction = selected_model.predict(processed_user_df)
            prediction_proba = selected_model.predict_proba(processed_user_df)[:, 1] # Probability of positive class

            if prediction[0] == 1:
                st.error(f"**Prediction: The individual is likely to have {disease_selection}.**")
            else:
                st.success(f"**Prediction: The individual is likely NOT to have {disease_selection}.**")

            st.write(f"Probability of {disease_selection}: **{prediction_proba[0]:.2f}**")

            # --- Store prediction in session state for "Past Predictions" tab ---
            st.session_state['predictions_db'].append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'user_id': st.session_state['user_id'],
                'username': st.session_state['username'],
                'disease': disease_selection,
                'model': selected_model_name,
                'input_data': user_input_data, # Store raw input
                'prediction': int(prediction[0]),
                'probability': float(prediction_proba[0]),
                'outcome_label': current_target_labels[int(prediction[0])]
            })
            # In a real Firestore setup, this would be a Firestore add_document call:
            # firestore_db.collection(f"artifacts/{app_id}/users/{st.session_state['user_id']}/predictions").add(prediction_data)


            st.markdown("---")
            st.subheader(f"Model Performance Visualizations (on {disease_selection} Test Set):")

            # --- Re-calculate metrics and plots for the selected model on the TEST set for display ---
            # This part is for displaying overall model performance in the app, not just single prediction.
            # It dynamically loads and preprocesses the full dataset again to get consistent test sets.

            # Load and preprocess the full dataset for evaluation display
            if disease_selection == "Diabetes":
                df_eval = pd.read_csv(current_data_url_full)
                cols_to_check_eval = current_cols_to_check_full
                for col in cols_to_check_eval:
                    df_eval[col] = df_eval[col].replace(0, np.nan)
                    df_eval[col] = df_eval[col].fillna(current_medians_modes[col])
                X_eval = df_eval.drop(current_target_col, axis=1)
                y_eval = df_eval[current_target_col]
            else: # Heart Disease
                df_eval = pd.read_csv(current_data_url_full, names=current_cols_to_check_full, na_values='?')
                for col in ['ca', 'thal']:
                    df_eval[col] = pd.to_numeric(df_eval[col], errors='coerce')
                    df_eval[col] = df_eval[col].fillna(current_medians_modes[col])
                df_eval[current_target_col] = df_eval[current_target_col].apply(lambda x: 1 if x > 0 else 0)
                categorical_cols_eval = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
                # MODIFICATION: Ensure full dataset categorical columns are float type before get_dummies
                for col in categorical_cols_eval:
                    if col in df_eval.columns:
                        df_eval[col] = df_eval[col].astype(float)
                df_eval = pd.get_dummies(df_eval, columns=categorical_cols_eval, drop_first=True, dtype=int)
                X_eval = df_eval.drop(current_target_col, axis=1)
                y_eval = df_eval[current_target_col]
                # Ensure X_eval columns match the training order
                X_eval = X_eval.reindex(columns=current_feature_cols_order, fill_value=0)


            X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(
                X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
            )
            
            # Apply scaling to X_test_app if Logistic Regression is selected for Heart Disease
            if disease_selection == "Heart Disease" and selected_model_name == "Logistic Regression":
                scaler_eval = StandardScaler()
                scaler_eval.fit(X_train_app) # Fit on training portion of eval data
                X_test_app_processed = scaler_eval.transform(X_test_app)
            else:
                X_test_app_processed = X_test_app


            y_pred_app = selected_model.predict(X_test_app_processed)
            y_prob_app = selected_model.predict_proba(X_test_app_processed)[:, 1]

            # Confusion Matrix
            cm_app = confusion_matrix(y_test_app, y_pred_app)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_app, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[current_target_labels[0], current_target_labels[1]],
                        yticklabels=[current_target_labels[0], current_target_labels[1]], ax=ax_cm)
            ax_cm.set_title(f'Confusion Matrix for {selected_model_name} ({disease_selection} Test Set)')
            ax_cm.set_ylabel('Actual Label')
            ax_cm.set_xlabel('Predicted Label')
            st.pyplot(fig_cm)

            # ROC Curve
            fpr_app, tpr_app, thresholds_app = roc_curve(y_test_app, y_prob_app)
            roc_auc_app = auc(fpr_app, tpr_app)
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(fpr_app, tpr_app, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_app:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'ROC Curve for {selected_model_name} ({disease_selection} Test Set)')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True)
            st.pyplot(fig_roc)

            # Feature Importance (if applicable)
            if hasattr(selected_model, 'feature_importances_'):
                # Use appropriate feature columns for the selected disease
                feature_importances_app = pd.Series(selected_model.feature_importances_, index=current_feature_cols_order).sort_values(ascending=False)

                fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
                sns.barplot(x=feature_importances_app.values, y=feature_importances_app.index, palette='viridis', hue=feature_importances_app.index, legend=False, ax=ax_fi)
                ax_fi.set_title(f'Feature Importances for {selected_model_name} ({disease_selection})')
                ax_fi.set_xlabel('Importance')
                ax_fi.set_ylabel('Feature')
                st.pyplot(fig_fi)


    with tab2:
        st.header(f"Batch Prediction via CSV Upload ({disease_selection})")
        st.write(f"Upload a CSV file containing patient health data for batch {disease_selection} prediction.")
        
        # Dynamically show required columns based on disease selection
        if disease_selection == "Diabetes":
            required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        else: # Heart Disease
            required_columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
        st.info(f"The CSV file should have the following columns: `{', '.join(required_columns)}`.")

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key=f"csv_uploader_{disease_selection}")

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success("CSV file uploaded successfully!")
                st.write("First 5 rows of uploaded data:")
                st.dataframe(batch_df.head())

                if not all(col in batch_df.columns for col in required_columns):
                    st.error("Error: The uploaded CSV is missing one or more required columns.")
                    st.write("Required columns:", required_columns)
                else:
                    # Preprocess the batch data based on selected disease
                    if disease_selection == "Diabetes":
                        processed_batch_df = current_preprocess_func(batch_df.copy())
                    else: # Heart Disease
                        processed_batch_df = current_preprocess_func(batch_df.copy())
                        # Apply scaling if Logistic Regression is selected for Heart Disease
                        if selected_model_name == "Logistic Regression":
                            scaler_batch = StandardScaler()
                            # Fit scaler on X_heart_full_scaler (from above) and transform batch input
                            # This assumes X_heart_full_scaler is consistent with training data
                            
                            # Re-load full heart disease data for scaler fitting (for consistency)
                            df_heart_full_scaler_batch = pd.read_csv(current_data_url_full, names=heart_column_names_full, na_values='?')
                            for col in ['ca', 'thal']:
                                df_heart_full_scaler_batch[col] = pd.to_numeric(df_heart_full_scaler_batch[col], errors='coerce')
                                mode_val = df_heart_full_scaler_batch[col].mode()[0]
                                df_heart_full_scaler_batch[col] = df_heart_full_scaler_batch[col].fillna(mode_val)
                            df_heart_full_scaler_batch['num'] = df_heart_full_scaler_batch['num'].apply(lambda x: 1 if x > 0 else 0)
                            categorical_cols_full_batch = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
                            for col in categorical_cols_full_batch:
                                if col in df_heart_full_scaler_batch.columns:
                                    df_heart_full_scaler_batch[col] = df_heart_full_scaler_batch[col].astype(float)
                            df_heart_full_scaler_batch = pd.get_dummies(df_heart_full_scaler_batch, columns=categorical_cols_full_batch, drop_first=True, dtype=int)
                            
                            X_heart_full_scaler_batch = df_heart_full_scaler_batch.drop('num', axis=1)
                            # Ensure X_heart_full_scaler_batch columns match the training order
                            X_heart_full_scaler_batch = X_heart_full_scaler_batch.reindex(columns=heart_feature_columns_order, fill_value=0)

                            scaler_batch.fit(X_heart_full_scaler_batch) # Fit scaler on the full dataset
                            processed_batch_df = pd.DataFrame(scaler_batch.transform(processed_batch_df), columns=processed_batch_df.columns)
                        
                    # Make batch predictions
                    batch_predictions = selected_model.predict(processed_batch_df)
                    batch_probabilities = selected_model.predict_proba(processed_batch_df)[:, 1]

                    # Add predictions and probabilities to the DataFrame
                    batch_df['Predicted_Outcome'] = batch_predictions
                    batch_df['Predicted_Probability'] = batch_probabilities.round(2)

                    st.subheader("Batch Prediction Results:")
                    st.dataframe(batch_df)

                    # Option to download results
                    csv_output = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_output,
                        file_name=f"{disease_selection.lower()}_batch_predictions.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
                st.info("Please ensure your CSV is correctly formatted with the specified columns.")
    
    with tab3:
        st.header("Past Predictions")
        st.write(f"Displaying past predictions for user: **{st.session_state['username']}** (ID: {st.session_state['user_id']})")

        if st.session_state['predictions_db']:
            # Convert list of prediction dicts to DataFrame for display
            predictions_df = pd.DataFrame(st.session_state['predictions_db'])
            
            # Reorder columns for better readability
            display_cols = ['timestamp', 'disease', 'model', 'prediction', 'probability', 'outcome_label', 'input_data']
            predictions_df = predictions_df[display_cols]

            st.dataframe(predictions_df)

            # Option to download past predictions
            csv_output_past = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Past Predictions as CSV",
                data=csv_output_past,
                file_name=f"{st.session_state['username']}_past_predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("No past predictions found. Make a prediction in the 'Single Patient Prediction' tab to see it here!")
