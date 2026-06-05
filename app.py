import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import plotly.express as px

st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

st.title("💳 Deep Learning Fraud Detection System")

model = tf.keras.models.load_model(
    "fraud_lstm_attention.keras",
    compile=False
)

scaler = pickle.load(
    open("scaler.pkl","rb")
)

uploaded_file = st.file_uploader(
    "Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    features = scaler.transform(df)

    sequence_length = 10

    X = []

    for i in range(len(features)-sequence_length):
        X.append(features[i:i+sequence_length])

    X = np.array(X)

    predictions = model.predict(X)

    fraud_prob = predictions.flatten()

    results = pd.DataFrame({
        "Fraud Probability": fraud_prob
    })

    st.subheader("Prediction Results")
    st.dataframe(results)

    high_risk = results[
        results["Fraud Probability"] > 0.5
    ]

    st.subheader("🚨 High Risk Transactions")

    st.dataframe(high_risk)

    fig = px.histogram(
        results,
        x="Fraud Probability",
        title="Fraud Probability Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Fraud Summary")

    st.metric(
        "High Risk Count",
        len(high_risk)
    )
