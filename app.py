import time

import streamlit as st

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# Page configuration


st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)





st.markdown("""
<style>
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeInUp 0.6s ease-out;
}
div[data-testid="stMetric"] {
    animation: fadeInUp 0.6s ease-out;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# Header section


st.markdown("""
# 📦 Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to:
- **Forecast freight costs accurately**
- **Detect risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
""")

st.divider()


# Sidebar


st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact**
- Improved cost forecasting
- Reduced invoice fraud & anomalies
- Faster finance operations
""")


# Freight Cost Prediction


if selected_model == "Freight Cost Prediction":
    st.subheader("🚚 Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Predict freight cost for a vendor invoice using **Invoice Dollars**
    to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity (for reference only — not used by current model)",
                min_value=1,
                value=1200
            )

        with col2:
            dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=18500.0
            )

        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:
        # The trained model only uses "Dollars" as a feature.
        input_data = {
            "Dollars": [dollars]
        }

        with st.spinner("Running freight cost model..."):
            time.sleep(0.6)  # brief pause so the spinner is visible
            prediction = predict_freight_cost(input_data)['Predicted_Freight']

        st.toast("Prediction completed!", icon="✅")

        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.success("Prediction completed successfully.")
        st.metric(
            label="Estimated Freight Cost",
            value=f"${prediction[0]:,.2f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.balloons()


# Invoice Flag Prediction


else:
    st.subheader("🔍 Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**
    Predict whether a vendor invoice should be **flagged for manual approval**
    based on abnormal cost, freight, or delivery patterns.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )
            freight = st.number_input(
                "Freight Cost",
                min_value=0.0,
                value=1.73
            )

        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )
            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button("Evaluate Invoice Risk")

    if submit_flag:
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        progress = st.progress(0, text="Preparing features...")
        for pct, label in [
            (30, "Scaling inputs..."),
            (65, "Running classifier..."),
            (100, "Finalizing result...")
        ]:
            time.sleep(0.3)
            progress.progress(pct, text=label)

        flag_prediction = predict_invoice_flag(input_data)['Predicted_flag']
        is_flagged = bool(flag_prediction[0])

        progress.empty()

        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        if is_flagged:
            st.error("⚠️ Invoice requires **MANUAL APPROVAL**")
            st.toast("Risk detected — flagged for review.", icon="⚠️")
            st.snow()
        else:
            st.success("✅ Invoice is **SAFE for Auto-Approval**")
            st.toast("No risk detected.", icon="✅")
            st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)