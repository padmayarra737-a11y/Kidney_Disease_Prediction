import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# Load Model & Preprocessor
# ===============================
model = joblib.load("artifacts/AdaBoost.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")
columns = joblib.load("artifacts/columns.pkl")

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="CKD Prediction",
    page_icon="🩺",
    layout="wide"
)

# ===============================
# Title
# ===============================
st.markdown(
    "<h1 style='text-align:center;color:#0d47a1;font-size:50px;'>🩺 Kidney Disease Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align:center;color:#555;'>Machine Learning Prediction </p>", unsafe_allow_html=True)

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.title("Enter Patient Details")

# Numeric inputs with sliders
age = st.sidebar.slider("Age", 1, 100, 50)
bp = st.sidebar.slider("Blood Pressure", 50, 180, 80)
sg = st.sidebar.selectbox("Specific Gravity", ["1.005", "1.010", "1.015", "1.020", "1.025"])
al = st.sidebar.slider("Albumin", 0, 5, 0)
su = st.sidebar.slider("Sugar", 0, 5, 0)
bgr = st.sidebar.slider("Blood Glucose Random", 50, 400, 120)
bu = st.sidebar.slider("Blood Urea", 1, 300, 40)
sc = st.sidebar.slider("Serum Creatinine", 0.1, 15.0, 1.2)
sod = st.sidebar.slider("Sodium", 100, 150, 140)
pot = st.sidebar.slider("Potassium", 2.0, 7.0, 4.5)
hemo = st.sidebar.slider("Hemoglobin", 3.0, 17.0, 15.0)
pcv = st.sidebar.slider("Packed Cell Volume", 20, 60, 45)
wbcc = st.sidebar.slider("White Blood Cell Count", 2000, 25000, 8000)
rbcc = st.sidebar.slider("Red Blood Cell Count", 2.0, 7.0, 5.0)

# Categorical inputs with dropdowns
rbc = st.sidebar.selectbox("Red Blood Cells", ["normal", "abnormal"])
pc = st.sidebar.selectbox("Pus Cell", ["normal", "abnormal"])
pcc = st.sidebar.selectbox("Pus Cell Clumps", ["present", "notpresent"])
ba = st.sidebar.selectbox("Bacteria", ["present", "notpresent"])
htn = st.sidebar.selectbox("Hypertension", ["yes", "no"])
dm = st.sidebar.selectbox("Diabetes Mellitus", ["yes", "no"])
cad = st.sidebar.selectbox("Coronary Artery Disease", ["yes", "no"])
appet = st.sidebar.selectbox("Appetite", ["good", "poor"])
pe = st.sidebar.selectbox("Pedal Edema", ["yes", "no"])
ane = st.sidebar.selectbox("Anemia", ["yes", "no"])

# ===============================
# Build Input DataFrame
# ===============================
input_dict = {
    "age": age, "bp": bp, "sg": sg, "al": al, "su": su, "bgr": bgr, "bu": bu, "sc": sc,
    "sod": sod, "pot": pot, "hemo": hemo, "pcv": pcv, "wbcc": wbcc, "rbcc": rbcc,
    "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba, "htn": htn, "dm": dm, "cad": cad,
    "appet": appet, "pe": pe, "ane": ane
}
input_df = pd.DataFrame([input_dict])

# ===============================
# Prediction
# ===============================
# Custom CSS for larger button
# Custom CSS for larger, bold, centered button
st.markdown(
    """
    <style>
    div.stButton > button {
        display: block;
        margin: center;              /* centers the button */
        padding: 20px 60px;        /* bigger height & width */
        font-size: 28px;           /* larger text */
        font-weight: bold;         /* bold text */
        border-radius: 12px;       /* rounded corners */
        background-color: #0d47a1; /* custom color */
        color: white;              /* text color */
    }
    div.stButton > button:hover {
        background-color: #08306b; /* darker hover effect */
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if st.button("🔍 Predict Disease"):
    transformed_input = preprocessor.transform(input_df)
    proba = model.predict_proba(transformed_input)[0]

    ckd_index = list(model.classes_).index(1)
    ckd_probability = proba[ckd_index]

    threshold = 0.4

    prediction = 1 if ckd_probability >= threshold else 0

    if prediction == 1:
        st.markdown(
            f"""
            <div style="background-color:#ffe6e6;padding:25px;border-radius:12px;text-align:center;">
                <h2 style="color:#b30000;">⚠️ CKD Detected</h2>
                <p style="font-size:18px;">Probability: {proba[1]:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(proba[1])
    else:
        st.markdown(
            f"""
            <div style="background-color:#e6ffe6;padding:25px;border-radius:12px;text-align:center;">
                <h2 style="color:#006600;">✅ No CKD</h2>
                <p style="font-size:18px;">Probability: {proba[0]:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(proba[0])



# ===============================
# Sidebar Info
# ===============================
st.sidebar.info(
"""
Model: AdaBoost Classifier  
Dataset: CKD Prediction  
Accuracy: ~98.7%  
Framework: Streamlit  
"""
)
