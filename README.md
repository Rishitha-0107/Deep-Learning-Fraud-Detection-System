# 💳 Deep Learning Fraud Detection System

## 📌 Project Overview

The Deep Learning Fraud Detection System is an AI-powered financial security application designed to identify potentially fraudulent credit card transactions using Deep Learning techniques.

The system analyzes sequential transaction patterns and predicts the probability of fraud using an LSTM (Long Short-Term Memory) network combined with an Attention mechanism.

The project also provides interactive visualizations, fraud risk analysis, transaction monitoring, and business intelligence insights through a Streamlit dashboard.

---

## 🎯 Project Objectives

* Detect fraudulent financial transactions.
* Analyze transaction sequences using Deep Learning.
* Compare traditional and sequence-based models.
* Understand transaction risk patterns.
* Visualize fraud probabilities and risk levels.
* Build an interactive fraud intelligence dashboard.

---

## 📂 Dataset

Dataset Used:

**Credit Card Fraud Detection Dataset**

The dataset contains anonymized credit card transactions labeled as:

* Legitimate Transactions (Class = 0)
* Fraudulent Transactions (Class = 1)

Features include:

* Time
* V1 to V28 (PCA-transformed features)
* Amount
* Class (Target Variable)

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Data Processing

* NumPy
* Pandas
* Scikit-Learn

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Dashboard Development

* Streamlit

---

## 🧠 Deep Learning Architecture

### Model A: Dense Neural Network

Input Features
↓
Dense Layer
↓
Dense Layer
↓
Output Layer

### Model B: LSTM Network

Input Sequence
↓
Embedding/Sequence Input
↓
LSTM Layer
↓
Dense Layer
↓
Output Layer

### Model C: LSTM + Attention (Final Model)

Transaction Sequence
↓
LSTM Layer
↓
Attention Layer
↓
Global Average Pooling
↓
Dense Layer
↓
Fraud Prediction

---


## 🚀 Dashboard Features

### Transaction Analysis

Upload transaction datasets and analyze:

* Total Predictions
* High-Risk Transactions
* Medium-Risk Transactions

### Fraud Risk Prediction

Outputs:

* Fraud Probability
* Risk Classification

Risk Levels:

* Low Risk
* Medium Risk
* High Risk

### Interactive Visualizations

Includes:

* Fraud Probability Histogram
* Risk Distribution Pie Chart
* Fraud Trend Analysis
* High-Risk Transaction Table

---

## 📈 Model Evaluation

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

The LSTM + Attention model captures sequential transaction behavior more effectively than traditional Dense models.

---

## 📁 Project Structure

```text
Deep-Learning-Fraud-Detection-System/
│
├── app.py
├── requirements.txt
├── fraud_lstm_attention.keras
├── scaler.pkl
├── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Deep-Learning-Fraud-Detection-System.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📊 Dashboard Workflow

```text
Upload Transaction CSV
           ↓
Data Validation
           ↓
Feature Scaling
           ↓
Sequence Generation
           ↓
LSTM + Attention Prediction
           ↓
Fraud Probability Analysis
           ↓
Risk Classification
           ↓
Visualization Dashboard
```

---

## 🌟 Future Enhancements

* Real-Time Fraud Detection
* Attention Weight Visualization
* Fraud Alert Notifications
* Explainable AI using SHAP
* PDF Fraud Analysis Reports
* Banking Transaction API Integration
* Multi-Class Fraud Categorization

---

## 💡 Key Learnings

* Deep Learning for Financial Security
* Sequential Transaction Modeling
* Attention Mechanisms
* Fraud Analytics
* Streamlit Dashboard Development
* Explainable AI Concepts

---

## 👨‍💻 Author

L. Rishitha

Data Science Student

Machine Learning | Deep Learning | NLP | Data Analytics

---

## 📜 License

This project is developed for educational and research purposes.
