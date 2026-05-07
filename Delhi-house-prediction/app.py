import streamlit as st
import pickle
import pandas as pd

# Load model and columns
model = pickle.load(open("delhi_house_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("Delhi House Price Prediction")

st.write("Enter house details")

# Inputs (use correct names)
area = st.number_input("Area (sq ft)", min_value=100)

bhk = st.number_input("BHK", min_value=1)

bathroom = st.number_input("Bathroom", min_value=1)

# Create input dataframe
input_data = pd.DataFrame({
    "Area": [area],
    "BHK": [bhk],
    "Bathroom": [bathroom]
})

# Match training columns
input_data = input_data.reindex(
    columns=model_columns,
    fill_value=0
)

# Predict
if st.button("Predict Price"):

    prediction = model.predict(input_data)

    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")