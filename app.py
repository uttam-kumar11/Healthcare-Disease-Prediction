import streamlit as st
import pandas as pd
import numpy as np
import joblib # For loading the saved models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split # Needed for re-splitting data for evaluation display

# --- 1. Load Trained Models ---
# Ensure these .pkl files are in the same directory as app.py
try:
    log_reg_model = joblib.load('log_reg_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    dt_model = joblib.load('dt_model.pkl')
    st.success("Machine Learning Models loaded successfully!")
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'log_reg_model.pkl', 'rf_model.pkl', and 'dt_model.pkl' are in the same directory as app.py.")
    st.stop() # Stop the app if models can't be loaded

# Define the models dictionary after loading them
models = {
    "Logistic Regression": log_reg_model,
    "Random Forest": rf_model,
    "Decision Tree": dt_model
}

# --- 2. Define Preprocessing Function (consistent with notebook) ---
# This function will handle zero values and imputation for new user input
def preprocess_input(input_df):
    cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_check:
        # Replace 0 with NaN
        input_df[col] = input_df[col].replace(0, np.nan)
        # Impute NaN with the median from the original training data (important for consistency)
        # These medians are hardcoded for simplicity, in a robust app, they'd be loaded.
        if col == 'Glucose':
            median_val = 117.0
        elif col == 'BloodPressure':
            median_val = 72.0
        elif col == 'SkinThickness':
            median_val = 29.0
        elif col == 'Insulin':
            median_val = 125.0
        elif col == 'BMI':
            median_val = 32.3
        else:
            # Fallback, though for this specific dataset, these are the only cols with 0s needing imputation
            median_val = input_df[col].median()

        input_df[col] = input_df[col].fillna(median_val)
    return input_df

# --- Streamlit App Layout ---
st.set_page_config(page_title="Disease Prediction App", layout="wide", initial_sidebar_state="expanded")

st.title("🏥 Healthcare Analytics: Disease Prediction")
st.markdown("---")

# Sidebar for model selection and general info
st.sidebar.header("App Configuration")
selected_model_name = st.sidebar.selectbox(
    "Select Machine Learning Model:",
    list(models.keys())
)
selected_model = models[selected_model_name]
st.sidebar.info(f"Currently selected model: **{selected_model_name}**")

st.sidebar.markdown("---")
st.sidebar.header("About This App")
st.sidebar.write("""
This application uses machine learning to predict the likelihood of diabetes based on various health parameters.
It's built with Streamlit for an interactive and user-friendly experience.
""")
st.sidebar.markdown("---")

# Use tabs for single prediction and batch prediction
tab1, tab2 = st.tabs(["Single Patient Prediction", "Batch Prediction (CSV)"])

with tab1:
    st.header("Predict Diabetes Risk for a Single Patient")

    # Input form for patient data
    st.subheader("Enter Patient Health Data:")
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1, key="preg_single")
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=120, key="glu_single")
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=122, value=70, key="bp_single")

    with col2:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20, key="st_single")
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=846, value=80, key="ins_single")
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=67.1, value=25.0, format="%.1f", key="bmi_single")

    with col3:
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.3, format="%.3f", key="dpf_single")
        age = st.number_input("Age (years)", min_value=21, max_value=81, value=30, key="age_single")

    # Create a DataFrame from user input
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    # Add a predict button
    predict_button = st.button("Predict Diabetes Risk", key="predict_single")

    # --- Prediction Logic for Single Patient ---
    if predict_button:
        st.subheader("Prediction Results:")
        # Preprocess the user input
        processed_user_data = preprocess_input(user_data.copy()) # Use .copy() to avoid SettingWithCopyWarning

        # Make prediction
        prediction = selected_model.predict(processed_user_data)
        prediction_proba = selected_model.predict_proba(processed_user_data)[:, 1] # Probability of diabetes

        if prediction[0] == 1:
            st.error(f"**Prediction: The individual is likely to have Diabetes.**")
        else:
            st.success(f"**Prediction: The individual is likely NOT to have Diabetes.**")

        st.write(f"Probability of Diabetes: **{prediction_proba[0]:.2f}**")

        st.markdown("---")
        st.subheader("Model Performance Visualizations (on Test Set):")

        # --- Re-calculate metrics and plots for the selected model on the TEST set for display ---
        # This part is for displaying overall model performance in the app, not just single prediction.
        # In a real app, you might pre-generate these or store them more efficiently.
        # For demonstration, we'll quickly re-do the data loading and splitting here.
        data_url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        df_full = pd.read_csv(data_url)
        cols_to_check_full = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df_full[cols_to_check_full] = df_full[cols_to_check_full].replace(0, np.nan)
        for col in cols_to_check_full:
            if col == 'Glucose':
                median_val = 117.0
            elif col == 'BloodPressure':
                median_val = 72.0
            elif col == 'SkinThickness':
                median_val = 29.0
            elif col == 'Insulin':
                median_val = 125.0
            elif col == 'BMI':
                median_val = 32.3
            else:
                median_val = df_full[col].median()
            df_full[col] = df_full[col].fillna(median_val)

        X_full = df_full.drop('Outcome', axis=1)
        y_full = df_full['Outcome']
        X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)


        y_pred_app = selected_model.predict(X_test_app)
        y_prob_app = selected_model.predict_proba(X_test_app)[:, 1]

        # Confusion Matrix
        cm_app = confusion_matrix(y_test_app, y_pred_app)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_app, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted No Diabetes', 'Predicted Diabetes'],
                    yticklabels=['Actual No Diabetes', 'Actual Diabetes'], ax=ax_cm)
        ax_cm.set_title(f'Confusion Matrix for {selected_model_name} (Test Set)')
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
        ax_roc.set_title(f'ROC Curve for {selected_model_name} (Test Set)')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True)
        st.pyplot(fig_roc)

        # Feature Importance (if applicable)
        if hasattr(selected_model, 'feature_importances_'):
            feature_importances_app = pd.Series(selected_model.feature_importances_, index=X_full.columns).sort_values(ascending=False)
            fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
            sns.barplot(x=feature_importances_app.values, y=feature_importances_app.index, palette='viridis', hue=feature_importances_app.index, legend=False, ax=ax_fi)
            ax_fi.set_title(f'Feature Importances for {selected_model_name}')
            ax_fi.set_xlabel('Importance')
            ax_fi.set_ylabel('Feature')
            st.pyplot(fig_fi)

with tab2:
    st.header("Batch Prediction via CSV Upload")
    st.write("Upload a CSV file containing patient health data for batch diabetes prediction.")
    st.info("The CSV file should have the following columns: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success("CSV file uploaded successfully!")
            st.write("First 5 rows of uploaded data:")
            st.dataframe(batch_df.head())

            # Check if all required columns are present
            required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            if not all(col in batch_df.columns for col in required_columns):
                st.error("Error: The uploaded CSV is missing one or more required columns.")
                st.write("Required columns:", required_columns)
            else:
                # Preprocess the batch data
                processed_batch_df = preprocess_input(batch_df.copy())

                # Make batch predictions
                batch_predictions = selected_model.predict(processed_batch_df)
                batch_probabilities = selected_model.predict_proba(processed_batch_df)[:, 1]

                # Add predictions and probabilities to the DataFrame
                batch_df['Predicted_Outcome'] = batch_predictions
                batch_df['Predicted_Probability_Diabetes'] = batch_probabilities.round(2)

                st.subheader("Batch Prediction Results:")
                st.dataframe(batch_df)

                # Option to download results
                csv_output = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_output,
                    file_name="diabetes_batch_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            st.info("Please ensure your CSV is correctly formatted with the specified columns.")

