import streamlit as st
import joblib
import pandas as pd
import base64
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Loanwise AI",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = joblib.load("svm_model.pkl")

# -------------------------------------------------
# BACKGROUND IMAGE
# -------------------------------------------------
def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_file = "loanbackground.jpg"
image_base64 = get_base64_image(image_file)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .card {{
        background-color: rgba(15, 23, 42, 0.93);
        padding: 40px;
        border-radius: 20px;
        max-width: 750px;
        margin: 50px auto;
        box-shadow: 0px 20px 45px rgba(0,0,0,0.7);
        color: white;
    }}

    .title {{
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }}

    .subtitle {{
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 30px;
    }}

    .approved {{
        background: linear-gradient(90deg, #16a34a, #22c55e);
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-top: 20px;
    }}

    .rejected {{
        background: linear-gradient(90deg, #dc2626, #ef4444);
        padding: 18px;
        border-radius: 12px;
        font-weight: bold;
        margin-top: 20px;
    }}

    .stButton>button {{
        width: 100%;
        height: 45px;
        border-radius: 10px;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# RESET FUNCTION
# -------------------------------------------------
def reset_fields():
    st.session_state.age = 0
    st.session_state.income = 0
    st.session_state.loan = 0
    st.session_state.duration = 1

# -------------------------------------------------
# CARD START
# -------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="title">Loanwise AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Loan Risk Assessment Platform</div>', unsafe_allow_html=True)

# -------------------------------------------------
# INPUTS (START EMPTY)
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, key="age")
    monthly_income = st.number_input("Monthly Income", min_value=0, key="income")

with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0, key="loan")
    payoff_duration = st.number_input("Payoff Duration (months)", min_value=1, key="duration")

# -------------------------------------------------
# BUTTONS
# -------------------------------------------------
colA, colB = st.columns(2)

with colA:
    predict_clicked = st.button("🚀 Assess Application")

with colB:
    st.button("🔄 Reset", on_click=reset_fields)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if predict_clicked:

    with st.spinner("Analyzing credit risk with AI..."):
        time.sleep(2)

    prediction = model.predict([[age, monthly_income, loan_amount, payoff_duration]])
    probability = model.decision_function([[age, monthly_income, loan_amount, payoff_duration]])
    confidence = round(abs(probability[0]) * 10, 2)

    if prediction[0] == 1:

        st.markdown(
            f'<div class="approved">✅ LOAN APPROVED<br>AI Confidence Score: {confidence}%</div>',
            unsafe_allow_html=True
        )

        # PDF CERTIFICATE
        pdf_file = "approval_certificate.pdf"
        doc = SimpleDocTemplate(pdf_file)
        elements = []

        style = ParagraphStyle(
            name='Normal',
            fontSize=14,
            textColor=colors.black
        )

        elements.append(Paragraph("FinTrust AI Loan Approval Certificate", style))
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(f"Applicant Age: {age}", style))
        elements.append(Paragraph(f"Monthly Income: {monthly_income}", style))
        elements.append(Paragraph(f"Loan Amount: {loan_amount}", style))
        elements.append(Paragraph("Status: APPROVED", style))

        doc.build(elements)

        with open(pdf_file, "rb") as file:
            st.download_button(
                label="📄 Download Approval Certificate",
                data=file,
                file_name="Loan_Approval_Certificate.pdf",
                mime="application/pdf"
            )

    else:
        reasons = []

        if age < 21:
            reasons.append("Applicant below minimum age requirement.")
        if monthly_income < 3000:
            reasons.append("Income below policy threshold.")
        if loan_amount > monthly_income * payoff_duration * 0.5:
            reasons.append("Loan exceeds safe debt-to-income ratio.")
        if payoff_duration > 48:
            reasons.append("Repayment period exceeds allowed maximum.")
        if len(reasons) == 0:
            reasons.append("Application failed internal AI risk model.")

        reasons_html = "<br>- " + "<br>- ".join(reasons)

        st.markdown(
            f'<div class="rejected">❌ LOAN REJECTED<br>AI Confidence Score: {confidence}% {reasons_html}</div>',
            unsafe_allow_html=True
        )

# -------------------------------------------------
# CHART
# -------------------------------------------------
st.markdown("### Financial Comparison")

data = pd.DataFrame({
    'Metric': ['Monthly Income', 'Loan Amount'],
    'Value': [monthly_income, loan_amount]
})

st.bar_chart(data.set_index('Metric'))

st.markdown('</div>', unsafe_allow_html=True)
