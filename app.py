import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load trained model pipeline
# -----------------------------
with open("pipeline.pkl", "rb") as f:
    model = pickle.load(f)  # make sure this is a pipeline including preprocessing

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🧠 Personality Classification App")
st.write("Predict your personality type using the trained ML model.")

st.sidebar.header("Enter Features")

# Inputs
time_spent = st.sidebar.number_input("Time Spent Alone", value=0.0)
social_attendance = st.sidebar.number_input("Social Event Attendance", value=0.0)
going_outside = st.sidebar.number_input("Going Outside", value=0.0)
friends_size = st.sidebar.number_input("Friends Circle Size", value=0.0)
post_frequency = st.sidebar.number_input("Post Frequency", value=0.0)
stage_fear = st.sidebar.selectbox("Stage Fear", ["Yes", "No"])
drained_after = st.sidebar.selectbox("Drained After Socializing", ["Yes", "No"])

# Prepare DataFrame
input_df = pd.DataFrame([{
    "Time_spent_Alone": time_spent,
    "Social_event_attendance": social_attendance,
    "Going_outside": going_outside,
    "Friends_circle_size": friends_size,
    "Post_frequency": post_frequency,
    "Stage_fear": stage_fear,
    "Drained_after_socializing": drained_after
}])

st.subheader("📥 Input Data")
st.write(input_df)

# Predict
if st.button("Predict Personality"):
    prediction = model.predict(input_df)[0]   # e.g., 0 or 1
    label_map = {1: "Introvert", 0: "Extrovert"}
    st.success(f"Predicted Personality: {label_map[prediction]}")