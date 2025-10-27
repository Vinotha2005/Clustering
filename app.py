import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------------------------------------
# Load Model, Scaler, and Feature Columns
# -------------------------------------------------------------
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# -------------------------------------------------------------
# Streamlit App UI
# -------------------------------------------------------------
st.set_page_config(page_title="Country Income Prediction", page_icon="🌍", layout="wide")

st.title("🌍 Country Income Prediction App")
st.markdown("Use this app to **predict the income** of a country based on its socio-economic features.")

st.divider()

# -------------------------------------------------------------
# Option 1: Manual input
# -------------------------------------------------------------
st.subheader("🔹 Manual Input")

col1, col2 = st.columns(2)
user_input = {}

for i, col in enumerate(columns):
    if i % 2 == 0:
        user_input[col] = col1.number_input(f"Enter {col}", value=0.0)
    else:
        user_input[col] = col2.number_input(f"Enter {col}", value=0.0)

if st.button("🚀 Predict Income"):
    input_df = pd.DataFrame([user_input])
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]
    st.success(f"✅ Predicted Income: **{prediction:.2f}**")

st.divider()

# -------------------------------------------------------------
# Option 2: CSV upload
# -------------------------------------------------------------
st.subheader("📂 Upload CSV File for Bulk Prediction")

uploaded_file = st.file_uploader("Upload a CSV file (with same columns as training data)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Drop 'country' column if exists
    if 'country' in df.columns:
        df = df.drop('country', axis=1)

    # Align columns to match model training
    df = df.reindex(columns=columns, fill_value=0)

    # Transform and predict
    scaled = scaler.transform(df)
    preds = model.predict(scaled)

    df['Predicted_Income'] = preds

    st.success("✅ Predictions generated successfully!")
    st.write("### Results Preview")
    st.dataframe(df.head())

    # Download predictions
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions as CSV", csv, "predicted_income.csv", "text/csv")
