import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Deep Learning Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>
.main-title{
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#1E88E5;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
}

.metric-card{
    background-color:#f8f9fa;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fraud_lstm_attention.keras",
        compile=False
    )

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

try:
    model = load_model()
    scaler = load_scaler()

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<p class="main-title">💳 Deep Learning Fraud Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">LSTM + Attention Based Transaction Risk Analysis Dashboard</p>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.header("📋 Instructions")

    st.write("""
    Upload a CSV containing transaction records.

    Expected Columns:

    - Time
    - V1 to V28
    - Amount

    Optional:
    - Class (will be ignored)

    The system will:
    - Predict fraud probability
    - Identify high-risk transactions
    - Visualize fraud scores
    """)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload Transaction CSV",
    type=["csv"]
)

if uploaded_file:

    try:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset Uploaded Successfully")

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        # ---------------------------------------------
        # Remove target column if exists
        # ---------------------------------------------

        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        # ---------------------------------------------
        # Validate Features
        # ---------------------------------------------

        expected_columns = [
            'Time'
        ] + [f'V{i}' for i in range(1, 29)] + ['Amount']

        missing_columns = [
            col for col in expected_columns
            if col not in df.columns
        ]

        if len(missing_columns) > 0:

            st.error(
                f"Missing Required Columns: {missing_columns}"
            )
            st.stop()

        # Keep correct order

        df = df[expected_columns]

        # ---------------------------------------------
        # Scaling
        # ---------------------------------------------

        scaled_features = scaler.transform(df)

        # ---------------------------------------------
        # Create Sequences
        # ---------------------------------------------

        sequence_length = 10

        sequences = []

        for i in range(
            len(scaled_features) - sequence_length
        ):
            sequences.append(
                scaled_features[
                    i:i+sequence_length
                ]
            )

        sequences = np.array(sequences)

        if len(sequences) == 0:

            st.error(
                "Need at least 10 transactions."
            )
            st.stop()

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        predictions = model.predict(
            sequences,
            verbose=0
        )

        fraud_scores = predictions.flatten()

        results = pd.DataFrame({
            "Transaction_ID":
                range(
                    sequence_length,
                    sequence_length + len(fraud_scores)
                ),
            "Fraud_Probability":
                fraud_scores
        })

        # ---------------------------------------------
        # Risk Labels
        # ---------------------------------------------

        def risk_label(score):

            if score >= 0.8:
                return "High Risk"

            elif score >= 0.5:
                return "Medium Risk"

            else:
                return "Low Risk"

        results["Risk_Level"] = (
            results["Fraud_Probability"]
            .apply(risk_label)
        )

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        high_risk = results[
            results["Risk_Level"] == "High Risk"
        ]

        medium_risk = results[
            results["Risk_Level"] == "Medium Risk"
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Predictions",
                len(results)
            )

        with col2:
            st.metric(
                "High Risk",
                len(high_risk)
            )

        with col3:
            st.metric(
                "Medium Risk",
                len(medium_risk)
            )

        st.divider()

        # ---------------------------------------------
        # Results Table
        # ---------------------------------------------

        st.subheader("📊 Fraud Prediction Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        # ---------------------------------------------
        # High Risk Transactions
        # ---------------------------------------------

        st.subheader(
            "🚨 High Risk Transactions"
        )

        if len(high_risk) > 0:

            st.dataframe(
                high_risk,
                use_container_width=True
            )

        else:

            st.success(
                "No High Risk Transactions Detected"
            )

        # ---------------------------------------------
        # Histogram
        # ---------------------------------------------

        st.subheader(
            "📈 Fraud Probability Distribution"
        )

        fig1 = px.histogram(
            results,
            x="Fraud_Probability",
            nbins=30
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # ---------------------------------------------
        # Pie Chart
        # ---------------------------------------------

        st.subheader(
            "📊 Risk Level Distribution"
        )

        pie_data = (
            results["Risk_Level"]
            .value_counts()
            .reset_index()
        )

        pie_data.columns = [
            "Risk",
            "Count"
        ]

        fig2 = px.pie(
            pie_data,
            values="Count",
            names="Risk"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ---------------------------------------------
        # Fraud Trend
        # ---------------------------------------------

        st.subheader(
            "📉 Fraud Probability Trend"
        )

        fig3 = go.Figure()

        fig3.add_trace(
            go.Scatter(
                y=results["Fraud_Probability"],
                mode="lines"
            )
        )

        fig3.update_layout(
            xaxis_title="Transaction",
            yaxis_title="Fraud Probability"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # ---------------------------------------------
        # Download Results
        # ---------------------------------------------

        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Processing Error: {str(e)}"
        )

else:

    st.info(
        "Upload a CSV file to begin fraud analysis."
    )
