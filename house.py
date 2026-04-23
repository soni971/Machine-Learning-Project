import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# ---------------- NIGHT BACKGROUND ----------------
st.markdown("""
<style>

/* Night House Background */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1600607687920-4e2a09cf159d");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Light overlay */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(255, 255, 255, 0.55);
    z-index: 0;
}

/* Main Card */
.main-card {
    position: relative;
    z-index: 1;
    background: rgba(255,255,255,0.97);
    padding: 50px;
    border-radius: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.3);
}

/* Heading */
h1 {
    color: #0b132b !important;
    font-size: 48px !important;
    font-weight: 900;
}

/* Subtitle */
.subtitle-text {
    font-size: 26px;
    font-weight: 700;
    color: #1c2541;
}

/* Performance Heading */
.performance-text {
    font-size: 32px;
    font-weight: 800;
    color: #0b132b;
}

/* BIG PREDICTED PRICE */
.prediction-box {
    margin-top: 30px;
    padding: 25px;
    background: #e0f2fe;
    border-radius: 15px;
    text-align: center;
    font-size: 36px;
    font-weight: 900;
    color: #0b132b;
    box-shadow: 0 5px 20px rgba(0,0,0,0.2);
}

/* Metric value */
[data-testid="stMetricValue"] {
    font-size: 38px !important;
    font-weight: 900 !important;
    color: #0b132b !important;
}

/* LIGHT SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e2e8f0, #cbd5e1);
    color: #000000;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #000000 !important;
    font-weight: 700;
}

section[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(90deg, #0b132b, #2563eb);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("house_price.csv")

X = df[['Size_sqft', 'Bedrooms', 'Age_years']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = r2_score(y_test, model.predict(X_test))

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏠 House Details")

size = st.sidebar.number_input("Size (sqft)", 500, 10000, 2500)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
age = st.sidebar.number_input("Age (years)", 0, 50, 5)

predict_btn = st.sidebar.button("Predict Price 💰")

# ---------------- MAIN CONTENT ----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("Smart House Price Prediction")
st.markdown('<div class="subtitle-text">Machine Learning based web application using Linear Regression.</div>', unsafe_allow_html=True)

if predict_btn:
    new_data = [[size, bedrooms, age]]
    prediction = model.predict(new_data)
    
    st.markdown(
        f'<div class="prediction-box">🏠 Predicted Price: ₹ {round(prediction[0],2)}</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="performance-text">📊 Model Performance</div>', unsafe_allow_html=True)
st.metric("R² Accuracy Score", round(accuracy, 3))

st.markdown('</div>', unsafe_allow_html=True)