import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

# Load models
price_model = joblib.load(open('price_model.pkl', 'rb'))
cluster_model = joblib.load(open('cluster_model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))

try:
    with open('cluster_names.json', 'r') as f:
        Cluster_names = json.load(f)
except:
    # Fallback if JSON missing
    Cluster_names = {0: "Affordable Small Diamonds", 1: "Mid-range Balanced Diamonds", 2: "Premium Heavy Diamonds"}

st.title("💎 Diamond Analytics: Price & Market Segment Predictor")

# Sidebar Inputs
st.sidebar.header("Diamond Attributes")
carat = st.sidebar.number_input("Carat Weight", 0.2, 5.0, 1.0)
cut = st.sidebar.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.sidebar.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.sidebar.selectbox("Clarity", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])
x = st.sidebar.number_input("X", 3.0, 10.0, 5.0)
y = st.sidebar.number_input("Y", 3.0, 10.0, 5.0)
z = st.sidebar.number_input("Z", 2.0, 6.0, 3.0)

cut_m = {"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}
color_m = {"D":0,"E":1,"F":2,"G":3,"H":4,"I":5,"J":6}
clarity_m = {"IF":0,"VVS1":1,"VVS2":2,"VS1":3,"VS2":4,"SI1":5,"SI2":6,"I1":7}

# Logic for carat_category
if carat < 0.5:
  c_cat_num = 0  #Light
elif carat <= 1.5:
  c_cat_num = 1  #Medium
else:
  c_cat_num = 2    #Heavy

volume = x*y*z
#price_per_carat = 0
dimension_ratio = (x+y)/(2*z)


input_df = np.array([[
        carat,
        cut_m[cut],
        color_m[color],
        clarity_m[clarity],
        x,
        y,
        z,
        volume,
      #  price_per_carat,
        dimension_ratio,
        c_cat_num
]])


if st.button("Predict Price"):
  try:
      # 1. Price Prediction
      price_pred = price_model.predict(input_df)[0]
      st.success(f"Estimated Price: ₹{price_pred:,.2f}")

      # 2. Cluster Prediction
      input_scaled = scaler.transform(input_df)
      cluster = cluster_model.predict(input_scaled)[0]

      st.info(f"Market Segment: {Cluster_names[cluster]}")
  except Exception as e:
      st.error(f"Prediction Error: {e}")
      st.warning("Ensure the model was trained with the same 12 features used here.")
