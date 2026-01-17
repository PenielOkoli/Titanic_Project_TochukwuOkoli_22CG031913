import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration (Must be the first line)
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Custom CSS for a beautiful look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    # Adjust path to look inside the 'model' folder
    model_path = os.path.join('model', 'titanic_survival_model.pkl')
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Please ensure 'titanic_survival_model.pkl' is in the 'model' folder.")
    st.stop()

# --- Header ---
st.title("🚢 Titanic Survival Prediction System")
st.markdown("Enter the passenger details below to predict their survival probability.")
st.write("---")

# --- Input Form ---
# We use columns to make the layout compact and clean
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = 1st, 2 = 2nd, 3 = 3rd")
    sex = st.selectbox("Gender", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)

with col2:
    fare = st.number_input("Fare ($)", min_value=0.0, value=15.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], 
                            format_func=lambda x: {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}[x])

# --- Prediction Logic ---
if st.button("Predict Survival Status"):
    # Create a DataFrame matching the model's training input
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    # Get prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of survival
    
    st.write("---")
    
    if prediction == 1:
        st.success(f"**Result: SURVIVED**")
        st.balloons()
        st.write(f"This passenger had a **{probability:.1%}** chance of survival.")
        

[Image of lifeboat titanic]

    else:
        st.error(f"**Result: DID NOT SURVIVE**")
        st.write(f"This passenger had a **{probability:.1%}** chance of survival.")
        

[Image of titanic sinking illustration]


# --- Footer ---
st.markdown("---")
st.caption("Project developed for Machine Learning Assessment. Model: Random Forest Classifier.")