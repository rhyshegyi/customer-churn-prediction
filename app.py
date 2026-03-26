"""
app.py — Streamlit Web Dashboard for Customer Churn Prediction.

Provides a live, interactive UI to make churn predictions using the
trained Gradient Boosting model and Pydantic validation layer.
"""
import streamlit as st
import os

# Set page config first
st.set_page_config(
        page_title="Churn Predictor",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

from src.predict import predict_churn
from pydantic import ValidationError

st.title("📡 Telco Customer Churn Predictor")
st.markdown(
    "Predict whether a customer is at risk of leaving based on their demographics, "
    "account details, and active services."
)

tab1, tab2 = st.tabs(["🔮 Make a Prediction", "📊 Model Insights"])

# ═══════════════════════════════════════════════════════════════════
# Tab 1: Predictor
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.header("Customer Profile")

    # Form to collect the 19 Pydantic fields
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
            partner = st.selectbox("Has Partner?", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

        with col2:
            st.subheader("Account Details")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing?", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

        with col3:
            st.subheader("Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Provider", ["Fiber optic", "DSL", "No"])
            
            # Helper to conditionally disable Internet options if "No" is selected
            inet_opt = ["Yes", "No", "No internet service"]
            idx = 2 if internet == "No" else 1 # Default to 'No' if they have internet, else 'No internet service'

            online_sec = st.selectbox("Online Security", inet_opt, index=idx)
            online_backup = st.selectbox("Online Backup", inet_opt, index=idx)
            device_prot = st.selectbox("Device Protection", inet_opt, index=idx)
            tech_support = st.selectbox("Tech Support", inet_opt, index=idx)
            stream_tv = st.selectbox("Streaming TV", inet_opt, index=idx)
            stream_movies = st.selectbox("Streaming Movies", inet_opt, index=idx)

        submit = st.form_submit_button("Predict Churn Risk", type="primary")

    if submit:
        # Build payload matching CustomerInput schema
        payload = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges)
        }

        try:
            # Call our prediction layer
            st.spinner("Analysing customer profile...")
            result = predict_churn(payload)
            
            # Display results
            st.divider()
            
            prob = result["churn_probability"] * 100
            prediction = result["prediction"]

            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if prediction == "Churn":
                    st.error("🚨 HIGH RISK")
                    st.metric(label="Churn Probability", value=f"{prob:.1f}%")
                else:
                    st.success("✅ RETAINED")
                    st.metric(label="Churn Probability", value=f"{prob:.1f}%")
            
            with col_res2:
                st.progress(result["churn_probability"])
                st.caption(f"Model ID: `{result['model_version']}`")
                if prediction == "Churn":
                    st.warning("Recommendation: Offer proactive discount or long-term contract upgrade.")

        except ValidationError as e:
            st.error("Validation Error — Data failed Pydantic schema checks.")
            st.json(e.errors())
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# Tab 2: Model Insights
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.header("How the Model Works")
    st.markdown(
        "This predictor uses a **Gradient Boosting Classifier** tuned with `GridSearchCV`. "
        "It achieved an **ROC-AUC of ~0.85** on the held-out test set."
    )

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    eval_plot = os.path.join(models_dir, "evaluation_plots.png")
    feat_plot = os.path.join(models_dir, "feature_importance.png")

    if os.path.exists(feat_plot):
        st.subheader("What drives churn?")
        st.markdown("The top 15 features the model uses to make its decision.")
        st.image(feat_plot, use_container_width=True)
    
    if os.path.exists(eval_plot):
        st.divider()
        st.subheader("Model Performance")
        st.markdown("Confusion matrix and ROC curve evaluated on 1,400 unseen customers.")
        st.image(eval_plot, use_container_width=True)
    
    if not os.path.exists(feat_plot) and not os.path.exists(eval_plot):
        st.info("Plots not found. Run `python src/evaluate.py` to generate them.")
