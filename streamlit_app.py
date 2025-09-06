import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title('MFI Loan Prediction App (Sri Lanka)')
# --- Helper Functions ---
def load_assets():
    """Loads the pre-trained model and label encoders."""
    try:
        with open('xgboost_MicroFinLoan_default_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('MicroFinlabel_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        st.error("Error: Model or label encoders not found. Please ensure 'xgboost_MicroFinLoan_default_model.pkl' and 'MicroFinlabel_encoders.pkl' are in the same directory.")
        return None, None

def predict_loan_default(model, encoders, data):
    """
    Predicts the loan default status based on the input data.
    Assumes a fixed threshold for classification.
    """
    # Map the UI labels to the model's feature names
    model_data = {
        'BorrowerAge': data['Age'],
        'HouseholdIncome': data['AnnualIncome'],
        'MicroLoanAmount': data['LoanAmount'],
        'RepaymentHistoryScore': data.get('CreditScore', 650),  # Assuming a default if not present in the new UI
        'MonthsInIncomeActivity': data['MonthsEmployed'],
        'OutstandingLoansCount': data.get('OutstandingLoansCount', 1), # Assuming a default if not present in the new UI
        'MicroLoanInterestRate': data['InterestRate'],
        'LoanTermMonths': data['LoanTerm'],
        'DebtToIncomeRatio': data.get('DebtToIncomeRatio', 0.5), # Assuming a default if not present in the new UI
        'EducationLevel': data['EducationLevel'],
        'PrimaryIncomeSource': data['EmploymentType'],
        'HouseholdStatus': data['MaritalStatus'],
        'OwnsLandOrHouse': data.get('OwnsLandOrHouse', 'No'), # Assuming a default if not present in the new UI
        'NumDependents': data['Dependents'],
        'MicroLoanPurpose': data['LoanPurpose'],
        'GroupLendingParticipation': data.get('GroupLendingParticipation', 'No') # Assuming a default if not present in the new UI
    }
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([model_data])

    # Apply the loaded label encoders to the categorical features
    categorical_features = ['EducationLevel', 'PrimaryIncomeSource', 'HouseholdStatus', 
                            'OwnsLandOrHouse', 'NumDependents', 'MicroLoanPurpose', 
                            'GroupLendingParticipation']
    
    # Check for and handle unknown categories before transformation
    for feature in categorical_features:
        if feature in encoders:
            le = encoders[feature]
            known_classes = list(le.classes_)
            if input_df[feature][0] not in known_classes:
                st.error(f"Error: The value '{input_df[feature][0]}' for '{feature}' is not a recognized category by the model. Please select a valid option from the dropdowns.")
                return None, None
            
            input_df[feature] = le.transform(input_df[feature])

    # Make a prediction (we need probabilities to apply the threshold)
    prediction_proba = model.predict_proba(input_df)[:, 1][0]
    
    # --- IMPORTANT: Set Your Optimized Threshold Here ---
    # Replace the placeholder value with the best threshold you found (e.g., 0.35)
    optimized_threshold = 0.35 
    
    if prediction_proba > optimized_threshold:
        prediction = 1 # Predicted to default
    else:
        prediction = 0 # Predicted not to default

    return prediction, prediction_proba

# --- Streamlit UI and Logic ---

# Initialize session state for the clear button
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

st.set_page_config(page_title="MFI Loan Approval", layout="centered")

st.title("MF LOAN APPROVAL")
st.markdown("Please fill out all required information for your loan application.")

# Load the model and encoders at the beginning
model, encoders = load_assets()

if model and encoders:
    # --- Input Form ---
    with st.form(key='loan_form'):
        st.header("Personal Information")

        # Create two columns for a cleaner layout
        col1, col2 = st.columns(2)

        # First Column for numerical features
        with col1:
            st.session_state.form_data['Age'] = st.number_input("Age *", min_value=18, max_value=100, value=st.session_state.form_data.get('Age', 30))
            st.session_state.form_data['EducationLevel'] = st.selectbox("Education Level *", options=['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], index=st.session_state.form_data.get('EducationLevel', 0))
            st.session_state.form_data['Dependents'] = st.selectbox("Dependents *", options=['Yes', 'No'], index=st.session_state.form_data.get('Dependents', 0))
            st.session_state.form_data['LoanAmount'] = st.number_input("Loan Amount (Rs) *", min_value=0, value=st.session_state.form_data.get('LoanAmount', 500000))
            st.session_state.form_data['LoanTerm'] = st.number_input("Loan Term (months) *", min_value=1, value=st.session_state.form_data.get('LoanTerm', 60))

        # Second Column for categorical features
        with col2:
            st.session_state.form_data['AnnualIncome'] = st.number_input("Annual Income (Rs) *", min_value=0, value=st.session_state.form_data.get('AnnualIncome', 500000))
            st.session_state.form_data['MaritalStatus'] = st.selectbox("Marital Status *", options=['Married', 'Single', 'Divorced'], index=st.session_state.form_data.get('MaritalStatus', 0))
            st.session_state.form_data['EmploymentType'] = st.selectbox("Employment Type *", options=['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], index=st.session_state.form_data.get('EmploymentType', 0))
            st.session_state.form_data['MonthsEmployed'] = st.number_input("Months Employed *", min_value=0, value=st.session_state.form_data.get('MonthsEmployed', 24))
            st.session_state.form_data['InterestRate'] = st.number_input("Interest Rate (%) *", min_value=0.0, max_value=100.0, value=st.session_state.form_data.get('InterestRate', 5.5))
            st.session_state.form_data['LoanPurpose'] = st.selectbox("Loan Purpose *", options=['Business', 'Education', 'Other', 'Auto', 'Emergency', 'Health', 'Home'], index=st.session_state.form_data.get('LoanPurpose', 0))

        st.markdown("---") # Add a horizontal line to separate sections
        col3, col4 = st.columns([1, 1])
        with col3:
            clear_button = st.form_submit_button("Clear All")
        with col4:
            submit_button = st.form_submit_button("Submit Application")

    # Clear button logic
    if clear_button:
        st.session_state.form_data = {}
        st.experimental_rerun()
    
    # --- Prediction Result Display ---
    if submit_button:
        st.subheader("Application Status")
        
        # Make the prediction
        prediction, probability = predict_loan_default(model, encoders, {
            'Age': st.session_state.form_data.get('Age'),
            'AnnualIncome': st.session_state.form_data.get('AnnualIncome'),
            'EducationLevel': st.session_state.form_data.get('EducationLevel'),
            'MaritalStatus': st.session_state.form_data.get('MaritalStatus'),
            'Dependents': st.session_state.form_data.get('Dependents'),
            'EmploymentType': st.session_state.form_data.get('EmploymentType'),
            'MonthsEmployed': st.session_state.form_data.get('MonthsEmployed'),
            'LoanAmount': st.session_state.form_data.get('LoanAmount'),
            'InterestRate': st.session_state.form_data.get('InterestRate'),
            'LoanTerm': st.session_state.form_data.get('LoanTerm'),
            'LoanPurpose': st.session_state.form_data.get('LoanPurpose')
        })
        
        # Only proceed if prediction was successful (i.e., not None)
        if prediction is not None:
            if prediction == 1:
                st.error(f"Prediction: This loan is likely to **Default**.")
                st.write("Based on the provided information, the loan application is **NOT APPROVED**.")
            else:
                st.success(f"Prediction: This loan is likely **Not to Default**.")
                st.write("Based on the provided information, the loan application is **APPROVED**.")

            # Show the full probability for detailed analysis
            with st.expander("Show Prediction Details"):
                st.write(f"Default Probability: **{probability:.2f}**")
                st.markdown("The model's probability score indicates the likelihood of a default.")

