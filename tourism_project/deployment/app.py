import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the
newly introduced **Wellness Tourism Package**, based on their profile and
their interaction with the sales pitch. Enter the customer details below
to get a prediction.
""")

st.subheader("Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox(
    "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox(
    "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
monthly_income = st.number_input(
    "Monthly Income", min_value=0.0, max_value=200000.0, value=20000.0, step=500.0
)
passport = st.selectbox("Holds a Valid Passport?", ["No", "Yes"])
own_car = st.selectbox("Owns a Car?", ["No", "Yes"])

st.subheader("Trip Preferences")
number_of_person_visiting = st.number_input(
    "Number of People Visiting", min_value=1, max_value=10, value=2, step=1
)
number_of_children_visiting = st.number_input(
    "Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0, step=1
)
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
number_of_trips = st.number_input(
    "Average Number of Trips per Year", min_value=0.0, max_value=25.0, value=2.0, step=1.0
)

st.subheader("Sales Interaction Details")
product_pitched = st.selectbox(
    "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)
duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0, step=1.0
)
number_of_followups = st.number_input(
    "Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0, step=1.0
)
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"This customer is likely to purchase the Wellness Tourism "
                   f"Package (probability: {probability:.1%}).")
    else:
        st.info(f"This customer is unlikely to purchase the Wellness Tourism "
                f"Package (probability: {probability:.1%}).")
