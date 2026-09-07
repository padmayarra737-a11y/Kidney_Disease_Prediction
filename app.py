import streamlit as st
import pandas as pd
import joblib


# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🩺",
    layout="wide"
)


# ===============================
# Custom CSS - Modern UI
# ===============================

st.markdown(
    """
    <style>

    .main {
        background-color:#f5f8fc;
    }

    .title {
        text-align:center;
        color:#0d47a1;
        font-size:42px;
        font-weight:700;
    }

    .subtitle {
        text-align:center;
        color:#555;
        font-size:18px;
    }


    div.stButton > button {

        width:100%;
        height:50px;
        border-radius:12px;
        background-color:#1565c0;
        color:white;
        font-size:18px;
        font-weight:bold;

    }


    div.stButton > button:hover {

        background-color:#0d47a1;

    }


    </style>
    """,
    unsafe_allow_html=True
)



# ===============================
# Load Model and Columns
# ===============================

@st.cache_resource
def load_files():

    model = joblib.load(
        "notebook/data/ExtraTreesClassifier.pkl"
    )

    columns = joblib.load(
        "notebook/data/columns.pkl"
    )

    return model, columns



model, columns = load_files()



# ===============================
# Header
# ===============================

st.markdown(
    "<div class='title'>🩺 Kidney Disease Prediction System</div>",
    unsafe_allow_html=True
)


st.markdown(
    "<div class='subtitle'>Machine Learning Prediction using Extra Trees Classifier</div>",
    unsafe_allow_html=True
)


st.divider()



# ===============================
# Input Features
# ===============================

st.subheader("Enter Patient Details")



col1, col2, col3 = st.columns(3)



# -------- Column 1 --------

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=50
    )


    bp = st.number_input(
        "Blood Pressure",
        value=80
    )


    sg = st.selectbox(
        "Specific Gravity",
        [1.005,1.010,1.015,1.020,1.025]
    )


    al = st.number_input(
        "Albumin",
        value=0
    )


    su = st.number_input(
        "Sugar",
        value=0
    )


    rbc = st.selectbox(
        "Red Blood Cells",
        [0,1]
    )


    pc = st.selectbox(
        "Pus Cell",
        [0,1]
    )


    pcc = st.selectbox(
        "Pus Cell Clumps",
        [0,1]
    )



# -------- Column 2 --------


with col2:


    ba = st.selectbox(
        "Bacteria",
        [0,1]
    )


    bgr = st.number_input(
        "Blood Glucose Random",
        value=120
    )


    bu = st.number_input(
        "Blood Urea",
        value=40
    )


    sc = st.number_input(
        "Serum Creatinine",
        value=1.2
    )


    sod = st.number_input(
        "Sodium",
        value=140
    )


    pot = st.number_input(
        "Potassium",
        value=4.5
    )


    hemo = st.number_input(
        "Hemoglobin",
        value=15.0
    )



# -------- Column 3 --------


with col3:


    pcv = st.number_input(
        "Packed Cell Volume",
        value=45
    )


    wbcc = st.number_input(
        "White Blood Cell Count",
        value=8000
    )


    rbcc = st.number_input(
        "Red Blood Cell Count",
        value=5.0
    )


    htn = st.selectbox(
        "Hypertension",
        [0,1]
    )


    dm = st.selectbox(
        "Diabetes Mellitus",
        [0,1]
    )


    cad = st.selectbox(
        "Coronary Artery Disease",
        [0,1]
    )


    appet = st.selectbox(
        "Appetite",
        [0,1]
    )



pe = st.selectbox(
    "Pedal Edema",
    [0,1]
)


ane = st.selectbox(
    "Anemia",
    [0,1]
)



st.divider()



# ===============================
# Prediction
# ===============================


if st.button("🔍 Predict Kidney Disease"):


    input_data = {

        "age":age,
        "bp":bp,
        "sg":sg,
        "al":al,
        "su":su,
        "rbc":rbc,
        "pc":pc,
        "pcc":pcc,
        "ba":ba,
        "bgr":bgr,
        "bu":bu,
        "sc":sc,
        "sod":sod,
        "pot":pot,
        "hemo":hemo,
        "pcv":pcv,
        "wbcc":wbcc,
        "rbcc":rbcc,
        "htn":htn,
        "dm":dm,
        "cad":cad,
        "appet":appet,
        "pe":pe,
        "ane":ane

    }



    input_df = pd.DataFrame(
        [input_data]
    )


    # IMPORTANT:
    # Keep exactly the same feature order
    # used during model training

    input_df = input_df[columns]



    # Prediction

    prediction = model.predict(
        input_df
    )


    probability = model.predict_proba(
        input_df
    )



    st.subheader("Prediction Result")



    if prediction[0] == 1:

        st.error(
            "⚠️ Kidney Disease Detected"
        )


        st.write(
            f"Risk Probability: {probability[0][1]*100:.2f}%"
        )


    else:

        st.success(
            "✅ No Kidney Disease Detected"
        )


        st.write(
            f"Healthy Probability: {probability[0][0]*100:.2f}%"
        )



# ===============================
# Sidebar
# ===============================

st.sidebar.title("About Model")


st.sidebar.info(
"""
Model:
Extra Trees Classifier

Dataset:
Kidney Disease Prediction

Preprocessing:
No Scaling Used

Framework:
Streamlit
"""
)









