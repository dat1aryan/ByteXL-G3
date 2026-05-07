import streamlit as st
import pickle
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Delhi House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = pickle.load(open("delhi_house_model.pkl", "rb"))
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found: delhi_house_model.pkl")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

@st.cache_resource
def load_columns():
    """Load the model columns"""
    try:
        columns = pickle.load(open("model_columns.pkl", "rb"))
        return columns
    except FileNotFoundError:
        st.error("❌ Columns file not found: model_columns.pkl")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading columns: {str(e)}")
        st.stop()

# Load model and columns
model = load_model()
model_columns = load_columns()

# UI
st.title("🏠 Delhi House Price Predictor")
st.markdown("---")
st.write("Enter house details to predict the estimated price")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

with col2:
    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

with col3:
    bathroom = st.number_input(
        "Bathroom",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

st.markdown("---")

# Create input dataframe
input_data = pd.DataFrame({
    "Area": [area],
    "BHK": [bhk],
    "Bathroom": [bathroom]
})

# Match training columns
try:
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )
except Exception as e:
    st.error(f"Error preparing input data: {str(e)}")
    st.stop()

# Predict button and results
if st.button("🔮 Predict Price", use_container_width=True):
    try:
        prediction = model.predict(input_data)
        predicted_price = prediction[0]
        
        # Display result
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Estimated Price",
                value=f"₹ {predicted_price:,.0f}",
                delta=None
            )
        
        with col2:
            st.info(f"**Input Details:**\n- Area: {area:,} sq ft\n- BHK: {bhk}\n- Bathrooms: {bathroom}")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Delhi House Price Model")
