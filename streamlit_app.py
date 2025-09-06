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
        'RepaymentHistoryScore': data.get('CreditScore', 650),
        'MonthsInIncomeActivity': data['MonthsEmployed'],
        'OutstandingLoansCount': data.get('OutstandingLoansCount', 1),
        'MicroLoanInterestRate': data['InterestRate'],
        'LoanTermMonths': data['LoanTerm'],
        'DebtToIncomeRatio': data.get('DebtToIncomeRatio', 0.5),
        'EducationLevel': data['EducationLevel'],
        'PrimaryIncomeSource': data['EmploymentType'],
        'HouseholdStatus': data['MaritalStatus'],
        'OwnsLandOrHouse': data.get('OwnsLandOrHouse', 'No'),
        'NumDependents': data['Dependents'],
        'MicroLoanPurpose': data['LoanPurpose'],
        'GroupLendingParticipation': data.get('GroupLendingParticipation', 'No')
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
    # The default XGBoost threshold is 0.5
    optimized_threshold = 0.35 
    
    if prediction_proba > optimized_threshold:
        prediction = 1 # Predicted to default
    else:
        prediction = 0 # Predicted not to default

    return prediction, prediction_proba

# --- Streamlit UI and Logic ---

# Set up page config
st.set_page_config(page_title="MFI Loan Approval", layout="centered")

# Initialize session state for the clear button
if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'Age': 30,
        'AnnualIncome': 500000,
        'EducationLevel': 'High School',
        'MaritalStatus': 'Married',
        'Dependents': 'Yes',
        'EmploymentType': 'Full-time',
        'MonthsEmployed': 24,
        'LoanAmount': 500000,
        'InterestRate': 5.5,
        'LoanTerm': 60,
        'LoanPurpose': 'Business'
    }

st.title("MF LOAN APPROVAL")
st.markdown("Please fill out all required information for your loan application.")

# Load the model and encoders at the beginning
model, encoders = load_assets()

if model and encoders:
    # --- Input Form ---
    with st.form(key='loan_form'):
        st.header("Personal Information")
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.form_data['Age'] = st.number_input("Age *", min_value=18, max_value=100, value=st.session_state.form_data['Age'], key='age_input')
            st.session_state.form_data['EducationLevel'] = st.selectbox("Education Level *", options=['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], index=['High School', 'Bachelor\'s', 'Master\'s', 'PhD'].index(st.session_state.form_data['EducationLevel']), key='edu_level_select')
            st.session_state.form_data['Dependents'] = st.selectbox("Dependents *", options=['Yes', 'No'], index=['Yes', 'No'].index(st.session_state.form_data['Dependents']), key='dependents_select')

        with col2:
            st.session_state.form_data['AnnualIncome'] = st.number_input("Annual Income (Rs) *", min_value=0, value=st.session_state.form_data['AnnualIncome'], key='annual_income_input')
            st.session_state.form_data['MaritalStatus'] = st.selectbox("Marital Status *", options=['Married', 'Single', 'Divorced'], index=['Married', 'Single', 'Divorced'].index(st.session_state.form_data['MaritalStatus']), key='marital_status_select')

        st.header("Employment Information")
        col3, col4 = st.columns(2)
        with col3:
            st.session_state.form_data['EmploymentType'] = st.selectbox("Employment Type *", options=['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], index=['Full-time', 'Part-time', 'Self-employed', 'Unemployed'].index(st.session_state.form_data['EmploymentType']), key='employment_type_select')
        with col4:
            st.session_state.form_data['MonthsEmployed'] = st.number_input("Months Employed *", min_value=0, value=st.session_state.form_data['MonthsEmployed'], key='months_employed_input')

        st.header("Loan Information")
        col5, col6 = st.columns(2)
        with col5:
            st.session_state.form_data['LoanAmount'] = st.number_input("Loan Amount (Rs) *", min_value=0, value=st.session_state.form_data['LoanAmount'], key='loan_amount_input')
            st.session_state.form_data['LoanTerm'] = st.number_input("Loan Term (months) *", min_value=1, value=st.session_state.form_data['LoanTerm'], key='loan_term_input')
        with col6:
            st.session_state.form_data['InterestRate'] = st.number_input("Interest Rate (%) *", min_value=0.0, max_value=100.0, value=st.session_state.form_data['InterestRate'], key='interest_rate_input')
            st.session_state.form_data['LoanPurpose'] = st.selectbox("Loan Purpose *", options=['Business', 'Education', 'Other', 'Auto', 'Emergency', 'Health', 'Home'], index=['Business', 'Education', 'Other', 'Auto', 'Emergency', 'Health', 'Home'].index(st.session_state.form_data['LoanPurpose']), key='loan_purpose_select')

        st.markdown("---") # Add a horizontal line to separate sections
        col7, col8 = st.columns([1, 1])
        with col7:
            clear_button_placeholder = st.empty()
        with col8:
            submit_button = st.form_submit_button("Submit Application")

    # Clear button logic outside the form
    if clear_button_placeholder.button("Clear All"):
        st.session_state.form_data = {
            'Age': 30, 'AnnualIncome': 500000, 'EducationLevel': 'High School', 
            'MaritalStatus': 'Married', 'Dependents': 'Yes', 'EmploymentType': 'Full-time', 
            'MonthsEmployed': 24, 'LoanAmount': 500000, 'InterestRate': 5.5, 
            'LoanTerm': 60, 'LoanPurpose': 'Business'
        }
        st.experimental_rerun()
    
    # --- Prediction Result Display ---
    if submit_button:
        st.subheader("Application Status")
        
        # Make the prediction
        prediction, probability = predict_loan_default(model, encoders, st.session_state.form_data)
        
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

