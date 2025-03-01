import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# Sidebar: Dataset Selection
# ---------------------------
st.sidebar.header("Navigation")
dataset_choice = st.sidebar.selectbox("Choose a dataset:", ["Reliance", "Tesla"])
page = st.sidebar.radio(
    "Go to",
    ("Home", "Data Processing", "Model Training", "Prediction", "Data Visualization")
)

# File paths for datasets and models
dataset_paths = {
    "Reliance": r"C:\Users\avinash\Downloads\my projects\machine learning projects\data\RELIANCE.NS.csv",
    "Tesla": r"C:\Users\avinash\Downloads\my projects\machine learning projects\data\TSLA.csv"
}

model_paths = {
    "Reliance": "model.pkl",
    "Tesla": "model1.pkl"
}

# ---------------------------
# Load the selected dataset
# ---------------------------
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data(dataset_paths[dataset_choice])

# ---------------------------
# HOME PAGE
# ---------------------------
if page == "Home":
    st.title("📈 Welcome to Stock Prediction App")
    st.write(f"Dataset Selected: {dataset_choice}")
    st.write("Navigate using the sidebar to explore data, train the model, and make predictions!")

# ---------------------------
# MODEL TRAINING PAGE
# ---------------------------
elif page == "Model Training":
    st.title(f"📈 Model Training - {dataset_choice}")

    # Prepare dataset for training
    df_cleaned = df.drop(columns=['Date']) if 'Date' in df.columns else df.copy()
    df_cleaned = df_cleaned.dropna()

    # Ensure only 4 features are used
    X = df_cleaned.iloc[:, :4].values
    y = df_cleaned.iloc[:, -1].values  # Last column is the target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_paths[dataset_choice])

    st.success(f"✅ Model trained successfully and saved as '{model_paths[dataset_choice]}'")

    # Model Performance
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    st.write("### Model Performance on Training Data")
    st.write(f"MAE: {mean_absolute_error(y_train, y_pred_train):,.3f}")
    st.write(f"MSE: {mean_squared_error(y_train, y_pred_train):,.3f}")

    st.write("### Model Performance on Test Data")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred_test):,.3f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred_test):,.3f}")

# ---------------------------
# PREDICTION PAGE
# ---------------------------
elif page == "Prediction":
    st.title(f"🔮 Stock Price Prediction - {dataset_choice}")

    # Load the trained model based on dataset selection
    model = joblib.load(model_paths[dataset_choice])

    # Input fields for user
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        feature1 = st.number_input("Feature 1", value=0.0, step=0.1, format="%.3f")
    with col2:
        feature2 = st.number_input("Feature 2", value=0.0, step=0.1, format="%.3f")
    with col3:
        feature3 = st.number_input("Feature 3", value=0.0, step=0.1, format="%.3f")
    with col4:
        feature4 = st.number_input("Feature 4", value=0.0, step=0.1, format="%.3f")

    # Prediction Button
    if st.button("Predict Stock Price"):
        user_input = np.array([[feature1, feature2, feature3, feature4]])  # Convert input to 2D array
        prediction = model.predict(user_input)  # Predict using the loaded model
        st.success(f"📌 Predicted Stock Price: {prediction[0]:,.3f}")

# ---------------------------
# DATA VISUALIZATION PAGE
# ---------------------------
elif page == "Data Visualization":
    st.title(f"📊 Data Visualization: {dataset_choice}")

    # Load trained model
    model = joblib.load(model_paths[dataset_choice])

    # Prepare data
    df_cleaned = df.drop(columns=['Date']) if 'Date' in df.columns else df.copy()
    df_cleaned = df_cleaned.dropna()
    X = df_cleaned.iloc[:, :4].values
    y_actual = df_cleaned.iloc[:, -1].values  # Last column is the actual stock price

    # Predict stock prices using trained model
    try:
        y_predicted = model.predict(X)
    except ValueError as e:
        st.error(f"Prediction Error: {e}")
        y_predicted = np.zeros_like(y_actual)

    # Plot Actual vs Predicted Stock Prices (Bar Chart)
    st.write("### Actual vs Predicted Stock Prices (Bar Chart)")

    # Limit to first 50 values to avoid clutter
    num_samples = min(50, len(y_actual))
    indices = np.arange(num_samples)  # X-axis labels

    fig, ax = plt.subplots(figsize=(12, 5))

    if len(y_predicted) == len(y_actual):
        width = 0.3  # Space between bars
        ax.bar(indices - width/2, y_actual[:num_samples], width=width, label="Actual Price", color="blue")
        ax.bar(indices + width/2, y_predicted[:num_samples], width=width, label="Predicted Price", color="orange")

        ax.set_xlabel("Index")
        ax.set_ylabel("Stock Price")
        ax.set_xticks(indices)
        ax.set_xticklabels(indices, rotation=90)
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("⚠ Mismatch in Actual vs Predicted Data Lengths!")