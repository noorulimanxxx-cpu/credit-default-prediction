import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide"
)

# Load model
model_data = joblib.load("random_forest_model.pkl")

model = model_data["model"]
features = model_data["features"]

# Title
st.title("💳 Credit Default Prediction")
st.write(
    "Machine Learning model for predicting whether a credit card customer "
    "is likely to default on their payment."
)

st.divider()

# Model information
st.subheader("Random Forest Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", "77.62%")

with col2:
    st.metric("Default Recall", "59%")

with col3:
    st.metric("Default F1-Score", "54%")

st.divider()

# Upload data
st.subheader("Upload Customer Data")

st.write(
    "Upload a CSV file containing customer information. "
    "The file should contain the same predictor columns used to train the model."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data")
    st.dataframe(data.head())

    # Check required columns
    missing_columns = [
        column for column in features
        if column not in data.columns
    ]

    if missing_columns:

        st.error(
            "The following required columns are missing:"
        )

        st.write(missing_columns)

    else:

        # Select model features
        X_new = data[features]

        # Make predictions
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)[:, 1]

        # Add results
        results = data.copy()

        results["Predicted Default"] = predictions
        results["Default Probability"] = probabilities

        results["Risk Level"] = results["Default Probability"].apply(
            lambda x:
                "High Risk" if x >= 0.5
                else "Low Risk"
        )

        st.subheader("Prediction Results")

        st.dataframe(results)

        # Summary
        total_customers = len(results)
        predicted_defaults = sum(predictions)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Customers Analyzed",
                total_customers
            )

        with col2:
            st.metric(
                "Predicted Defaulters",
                predicted_defaults
            )

        # Download results
        csv = results.to_csv(index=False)

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="credit_default_predictions.csv",
            mime="text/csv"
        )