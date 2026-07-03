import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Smart Agriculture System", layout="wide")

# ---------------------- LOAD MODELS ----------------------
soil_model = joblib.load("soil_score.pkl")
crop_model = joblib.load("crop_model.pkl")
crop_scaler = joblib.load("crop_scaler.pkl")
crop_label_encoder = joblib.load("crop_label_encoder.pkl")
fertilizer_model = joblib.load("fertiliser.pkl")

# ---------------------- TITLE ----------------------
st.title("🌱 Smart Agriculture Prediction System")

# ---------------------- SESSION ----------------------
if "soil_result" not in st.session_state:
    st.session_state.soil_result = None
    st.session_state.soil_category = None
    st.session_state.crop_result = None
    st.session_state.fertilizer_result = None
    st.session_state.N = 0
    st.session_state.P = 0
    st.session_state.K = 0
    st.session_state.ph = 0
    st.session_state.rainfall = 0
    st.session_state.city = ""
    st.session_state.farmer_name = ""

# ---------------------- FARMER DETAILS ----------------------
st.subheader("👨‍🌾 Farmer Details")
col1, col2 = st.columns(2)
with col1:
    farmer_name = st.text_input("Farmer Name")
    st.session_state.farmer_name = farmer_name
with col2:
    city = st.text_input("City")
    st.session_state.city = city

# ---------------------- FERTILIZER MAP ----------------------
fertilizer_mapping = {
0:"Urea",1:"DAP",2:"MOP",3:"NPK 10-26-26",4:"NPK 20-20-20",
5:"NPK 17-17-17",6:"Compost",7:"Vermicompost",8:"Ammonium Sulphate",
9:"Single Super Phosphate",10:"Potassium Nitrate",11:"Calcium Nitrate",
12:"Zinc Sulphate",13:"Magnesium Sulphate",14:"Bone Meal",
15:"Neem Cake",16:"Gypsum",17:"Rock Phosphate",
18:"Organic Mix",19:"Bio Fertilizer"
}

# ---------------------- MENU ----------------------
menu = st.sidebar.selectbox(
    "Select Module",
    ["Soil Health Prediction", "Crop Prediction", "Fertilizer Recommendation"]
)

# ---------------------- INPUT ----------------------
def input_layout():
    c1, c2, c3, c4 = st.columns(4)
    with c1: N = st.number_input("N", 0.0, 200.0)
    with c2: P = st.number_input("P", 0.0, 200.0)
    with c3: K = st.number_input("K", 0.0, 200.0)
    with c4: ph = st.number_input("pH", 0.0, 14.0)

    c5, c6, c7 = st.columns(3)
    with c5: temperature = st.number_input("Temperature")
    with c6: humidity = st.number_input("Humidity")
    with c7: rainfall = st.number_input("Rainfall")

    return N, P, K, ph, temperature, humidity, rainfall

# ---------------------- SOIL ----------------------
if menu == "Soil Health Prediction":

    st.header("Soil Health Score Prediction")
    N, P, K, ph, temperature, humidity, rainfall = input_layout()

    if st.button("Predict Soil Health"):

        df = pd.DataFrame([{
            "N":N,"P":P,"K":K,
            "temperature":temperature,
            "humidity":humidity,
            "ph":ph,"rainfall":rainfall
        }])

        df["N_norm"]=df["N"]/140
        df["P_norm"]=df["P"]/145
        df["K_norm"]=df["K"]/205
        df["pH_norm"]=df["ph"]/14

        df = df[["N","P","K","temperature","humidity","ph","rainfall",
                 "N_norm","P_norm","K_norm","pH_norm"]]

        pred = soil_model.predict(df)[0]

        if pred < 40: cat="Poor"
        elif pred < 70: cat="Moderate"
        else: cat="Good"

        st.session_state.soil_result = round(pred,2)
        st.session_state.soil_category = cat
        st.session_state.N = N
        st.session_state.P = P
        st.session_state.K = K
        st.session_state.ph = ph
        st.session_state.rainfall = rainfall

        st.success(f"Soil Score: {pred:.2f} | Category: {cat}")

# ---------------------- CROP ----------------------
elif menu == "Crop Prediction":

    st.header("Crop Prediction")
    N, P, K, ph, temperature, humidity, rainfall = input_layout()

    if st.button("Predict Crop"):

        df = pd.DataFrame([{
            "N":N,"P":P,"K":K,
            "temperature":temperature,
            "humidity":humidity,
            "ph":ph,"rainfall":rainfall
        }])

        df["N_P_ratio"]=df["N"]/(df["P"]+1)
        df["K_N_ratio"]=df["K"]/(df["N"]+1)
        df["rainfall_ph"]=df["rainfall"]*df["ph"]

        scaled = crop_scaler.transform(df)
        pred = crop_model.predict(scaled)
        crop = crop_label_encoder.inverse_transform(pred)[0]

        st.session_state.crop_result = crop
        st.success(f"🌾 Recommended Crop: {crop}")

# ---------------------- FERTILIZER ----------------------
elif menu == "Fertilizer Recommendation":

    st.header("Fertilizer Recommendation")
    N, P, K, ph, temperature, humidity, rainfall = input_layout()

    if st.button("Recommend Fertilizer"):

        df = pd.DataFrame([{
            "N":N,"P":P,"K":K,
            "temperature":temperature,
            "humidity":humidity,
            "ph":ph,"rainfall":rainfall
        }])

        pred = fertilizer_model.predict(df)[0]
        fert = fertilizer_mapping.get(pred,"Unknown")

        st.session_state.fertilizer_result = fert
        st.success(f"🧪 Recommended Fertilizer: {fert}")

# ---------------------- MAP ----------------------
st.subheader("Farm Location Map")
st.map({"lat":[18.5204],"lon":[73.8567]})

# ---------------------- PDF REPORT ----------------------
st.sidebar.subheader("📄 Generate Final Report")

if st.sidebar.button("Generate PDF Report"):

    soil_score = st.session_state.soil_result or 0
    soil_category = st.session_state.soil_category or "N/A"
    crop_result = st.session_state.crop_result or "N/A"
    fertilizer_result = st.session_state.fertilizer_result or "N/A"

    N = st.session_state.N
    P = st.session_state.P
    K = st.session_state.K
    ph = st.session_state.ph
    rainfall = st.session_state.rainfall

    farmer_name = st.session_state.farmer_name
    city = st.session_state.city

    # -------- STYLES --------
    styles = getSampleStyleSheet()
    center_style = ParagraphStyle(name="Center", alignment=1, fontSize=16, spaceAfter=10)
    sub_center = ParagraphStyle(name="SubCenter", alignment=1, fontSize=12, spaceAfter=10)

    # -------- INTERPRETATION --------
    explanation = f"""
Soil condition is {soil_category}.

Nitrogen (N): {N}. {'Low, add Urea fertilizer.' if N < 50 else 'Good level.'}
Phosphorus (P): {P}. {'Low, use DAP fertilizer.' if P < 40 else 'Good level.'}
Potassium (K): {K}. {'Low, add Potash fertilizer.' if K < 50 else 'Good level.'}

Soil pH: {ph}. {'Acidic soil, add lime.' if ph < 6 else 'Suitable soil.' if ph <= 7.5 else 'Alkaline soil, use gypsum.'}

Rainfall: {rainfall}. {'Low, irrigation needed.' if rainfall < 150 else 'Sufficient rainfall.'}

Recommended Crop: {crop_result}.
Recommended Fertilizer: {fertilizer_result}.
"""

    # -------- PDF --------
    doc = SimpleDocTemplate("Smart_Agriculture_Report.pdf")
    elements = []

    elements.append(Paragraph("FERGUSSON COLLEGE", center_style))
    elements.append(Paragraph("Department of Computer Science", sub_center))
    elements.append(Spacer(1,20))

    elements.append(Paragraph("SMART AGRICULTURE REPORT", center_style))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Farmer Name: {farmer_name}", styles["Normal"]))
    elements.append(Paragraph(f"City: {city}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {datetime.now()}", styles["Normal"]))
    elements.append(Spacer(1,20))

    table = Table([
        ["Soil Score", soil_score],
        ["Soil Category", soil_category],
        ["Crop", crop_result],
        ["Fertilizer", fertilizer_result]
    ])
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),1,colors.black)]))
    elements.append(table)
    elements.append(Spacer(1,20))

    elements.append(Paragraph(explanation, styles["Normal"]))

    doc.build(elements)

    with open("Smart_Agriculture_Report.pdf","rb") as f:
        st.sidebar.download_button(
            "📥 Download PDF",
            data=f,
            file_name="Smart_Agriculture_Report.pdf",
            mime="application/pdf"
        )