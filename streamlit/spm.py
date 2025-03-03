import streamlit as st
import pickle

# Load the trained model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Function to add a background image and custom styles
def add_bg_and_styles():
    bg_url = "https://th.bing.com/th/id/OIP.M2dwMsL_3VMoGd3ILu5IZQHaD0?w=342&h=179&c=7&r=0&o=5&dpr=1.3&pid=1.7"
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
            background-position: center;
            color: #ffffff; /* White text */
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: #f5a623; /* Orange-golden text */
        }}

        .stTextArea textarea {{
            background-color: rgba(0, 0, 0, 0.7); /* Dark transparent background */
            color: #ffffff; /* White text */
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #f5a623; /* Orange border */
        }}

        .stButton>button {{
            background-color: #f5a623; /* Orange-golden button */
            color: black;
            font-weight: bold;
            border-radius: 8px;
            padding: 8px 15px;
        }}

        .stButton>button:hover {{
            background-color: #ffcc33; /* Lighter orange on hover */
            color: black;
        }}

        .stSidebar {{
            background-color: rgba(0, 0, 0, 0.6); /* Dark semi-transparent sidebar */
        }}

        /* Modify the color of displayed messages */
        .stAlert {{
            background-color: rgba(0, 0, 0, 0.8) !important; /* Dark background for alerts */
            color: white !important; /* White text */
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply styles
add_bg_and_styles()

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("🔍 Enter a message below to check if it's **Spam** or **Not Spam**.")

# User Input
user_input = st.text_area("Enter your message here:", "")

if st.button("Detect"):  # Changed button text to "Detect"
    if user_input:
        # Transform input using loaded vectorizer
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)

        # Display Result with dark background
        if prediction[0] == 1:
            st.error("🚨 **SPAM DETECTED!**", icon="🚨")
        else:
            st.success("✅ **This message is SAFE.**", icon="✅")
    else:
        st.warning("⚠ **Please enter a message to classify.**", icon="⚠")

# Sidebar Info
st.sidebar.subheader("📊 Model Loaded Successfully!")
