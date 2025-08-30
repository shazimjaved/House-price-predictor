import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ Load model and feature order from saved file (fail gracefully if missing)
try:
    model, feature_order = joblib.load('price_model.pkl')

except Exception as e:
    st.error("Model file 'price_model.pkl' not found or unreadable. Please train the model by running 'app.py' with 'zameen.csv' present to generate it.")
    st.stop()

# ✅ Manual encodings (must match training LabelEncoder mappings)
property_type_map = {'Flat': 0, 'House': 1, 'Upper Portion': 2, 'Lower Portion': 3, 'Farm House': 4}
purpose_map = {'For Sale': 1, 'For Rent': 0}
province_map = {'Islamabad Capital': 0, 'Punjab': 1, 'Sindh': 2, 'Khyber Pakhtunkhwa': 3, 'Balochistan': 4}
city_map = {'Islamabad': 0, 'Lahore': 1, 'Karachi': 2, 'Peshawar': 3, 'Quetta': 4 ,'Faisalabad':5 ,'Rawalpindi':6}  # Extend if needed
location_map = {'G-10': 0, 'E-11': 1, 'G-15': 2, 'Bani Gala': 3, 'DHA Valley': 4 ,'Samanzar Colony':5,'Shahra-e-Liaquat':6, 'Sundar':7,'Samanabad':8}  # Extend if needed

# ✅ Area conversion function
def convert_to_marla(size, unit):
    unit = unit.strip().lower()
    try:
        size = float(size)
    except:
        return None

    if unit == "marla":
        return size
    elif unit == "kanal":
        return size * 20
    elif unit in ["sq. yards", "square yards"]:
        return size / 30.25
    elif unit in ["sq. ft.", "square feet"]:
        return size / 225
    else:
        return None

# ✅ Streamlit UI
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction Model")
st.markdown("Made with &hearts; By Shazim Javed")

# Inputs
st.header("Enter Property Details:")

property_type = st.selectbox("Property Type", list(property_type_map.keys()))
purpose = st.selectbox("Purpose", list(purpose_map.keys()))
province = st.selectbox("Province", list(province_map.keys()))
city = st.selectbox("City", list(city_map.keys()))
location = st.selectbox("Location", list(location_map.keys()))

latitude = st.number_input("Latitude", value=33.7)
longitude = st.number_input("Longitude", value=73.0)

col1, col2 = st.columns(2)
with col1:
    baths = st.slider("Bathrooms", 1, 10, 2)
with col2:
    bedrooms = st.slider("Bedrooms", 1, 10, 3)

st.subheader("Area Information")
area_size = st.number_input("Area Size", value=5.0)
area_unit = st.selectbox("Area Unit", ["Marla", "Kanal", "Sq. Yards", "Sq. Ft."])

st.subheader("Date Property Was Listed")
year_added = st.number_input("Year Added", min_value=2015, max_value=2025, value=2023)
month_added = st.slider("Month", 1, 12, 6)
day_added = st.slider("Day", 1, 31, 15)

# ✅ Convert and encode input
total_area = convert_to_marla(area_size, area_unit)

if total_area is None:
    st.error("⚠️ Invalid area size/unit.")
else:
    # Add a prediction button
    if st.button("🚀 Enter to Predict", type="primary", use_container_width=True):
        # Show loading spinner and progress
        with st.spinner("🔮 Predicting price... Please wait..."):
            # Progress bar for visual feedback
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps for better UX
            status_text.text("📊 Processing property details...")
            progress_bar.progress(25)
            
            encoded_data = {
                'property_type': property_type_map[property_type],
                'location': location_map[location],
                'city': city_map[city],
                'province_name': province_map[province],
                'purpose': purpose_map[purpose],
                'latitude': latitude,
                'longitude': longitude,
                'baths': baths,
                'bedrooms': bedrooms,
                'year_added': year_added,
                'month_added': month_added,
                'day_added': day_added,
                'total_area_marla': total_area
            }
            
            status_text.text("🤖 Running AI prediction model...")
            progress_bar.progress(75)
            
            input_df = pd.DataFrame([encoded_data])
            input_df = input_df[feature_order]  # ✅ Ensure order matches training

            # ✅ Predict and show result
            log_price = model.predict(input_df)[0]
            actual_price = np.expm1(log_price)  # Reverse log1p
            
            status_text.text("✨ Finalizing results...")
            progress_bar.progress(100)

        # Clear loading elements
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"💰 **Estimated Price:** Rs. {actual_price:,.0f}")
    else:
        st.info("Click the button above to get the Estimated Price")