import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Background Image
background_image = "https://th.bing.com/th/id/OIP.l2e2w-1oUNDcngYmLtIoYgHaEr?w=270&h=184&c=7&r= &o=5&dpr=1.3&pid=1.7"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("{background_image}") no-repeat center center fixed;
        background-size: cover;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #B22222;  /* Dark Red Headings */
        text-shadow: 1px 1px 3px black;
    }}
    .stText, .stNumberInput, .stButton, .stRadio, .stSelectbox, .stDataFrame {{
        color: white !important;  /* White Text for Inputs */
        text-shadow: 1px 1px 2px black;
    }}
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: 2px solid white;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = "model1.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()

st.title("❤️ Heart Disease Risk Predictor")
st.sidebar.title("🔍 Navigation")
option = st.sidebar.radio("Choose a section", ["Prediction", "Dashboard"])

if option == "Prediction":
    st.header("📊 Enter Details for Prediction")
    age = st.number_input("Age")
    resting_rate = st.number_input("Resting Heart Rate")
    bp = st.number_input("Blood Pressure")
    cholesterol = st.number_input("Cholesterol Level")

    if st.button("Predict Risk Level"):
        user_input = np.array([[age, resting_rate, bp, cholesterol]])
        prediction = model.predict(user_input)[0]
        st.success(f"Your heart disease risk level is: {prediction}")

elif option == "Dashboard":
    st.header("📈 Data Insights & Visualizations")

    data = pd.DataFrame({
        "Age": np.random.randint(20, 80, 100),
        "Resting Rate": np.random.randint(50, 120, 100),
        "BP": np.random.randint(90, 180, 100),
        "Cholesterol": np.random.randint(150, 300, 100),
        "Risk Level": np.random.choice(["Low", "Medium", "High"], 100)
    })

    st.write("### 🏥 Summary Statistics")
    st.write(data.describe())

    st.write("### 📊 Risk Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=data["Risk Level"], palette="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### 📌 Feature Distributions")
    fig, ax = plt.subplots()
    sns.histplot(data["Age"], bins=20, kde=True, ax=ax, color="blue")
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x=data["Cholesterol"], ax=ax, color="skyblue")
    ax.set_title("Cholesterol Levels")
    st.pyplot(fig)
