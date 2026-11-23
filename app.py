
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("creditguard_model_v2.pkl")

try:
    package = load_model()

    model = package["model"]
    scaler = package["scaler"]
    threshold = package["threshold"]
    features = package["features"]

except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# ---------------- HEADER ----------------

st.title("🛡️ CreditGuard AI")
st.subheader("Intelligent Credit Card Fraud Detection")
st.caption("Machine Learning Based Transaction Risk Analysis")

st.divider()


# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Page",
    ["Dashboard", "Transaction Analysis", "About"]
)


# ---------------- PREDICTION FUNCTION ----------------

def analyze_transactions(data):

    missing = set(features) - set(data.columns)

    if missing:
        raise ValueError(
            "Missing required features: "
            + ", ".join(sorted(missing))
        )

    X = data[features].copy()

    X[["Time", "Amount"]] = scaler.transform(
        X[["Time", "Amount"]]
    )

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = data.copy()

    result["Fraud_Probability"] = probabilities * 100

    result["Prediction"] = np.where(
        predictions == 1,
        "Fraud",
        "Genuine"
    )

    result["Risk_Level"] = np.select(
        [
            probabilities >= 0.70,
            probabilities >= 0.30
        ],
        ["High", "Medium"],
        default="Low"
    )

    result["Alert"] = np.where(
        predictions == 1,
        "Fraud Alert - Review",
        np.where(
            result["Risk_Level"] == "Medium",
            "Additional Verification",
            "No Alert"
        )
    )

    return result, probabilities, predictions


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.header("Fraud Detection Overview")

    st.info(
        "Upload a transaction CSV in Transaction Analysis "
        "to generate fraud detection results."
    )

    st.markdown("### System Capabilities")

    c1, c2, c3 = st.columns(3)

    c1.metric("ML Models", "3")
    c2.metric("Prediction Type", "Binary")
    c3.metric("Analysis", "Batch CSV")

    st.markdown("### Detection Workflow")

    st.write(
        "Transaction Data → Preprocessing → "
        "ML Prediction → Risk Classification → Fraud Alert"
    )

    st.markdown("### Technologies Used")

    st.write(
        "Python | Pandas | NumPy | Scikit-learn | "
        "XGBoost | Streamlit"
    )


# ---------------- TRANSACTION ANALYSIS ----------------

elif page == "Transaction Analysis":

    st.header("Transaction Analysis")

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)

            # Preserve actual labels for optional evaluation
            has_labels = "Class" in data.columns

            if has_labels:
                actual = pd.to_numeric(
                    data["Class"],
                    errors="coerce"
                )

            input_data = data.drop(
                columns=["Class"],
                errors="ignore"
            )

            st.markdown("### Uploaded Dataset Preview")

            st.dataframe(
                data.head(),
                use_container_width=True
            )

            if st.button(
                "Analyze Transactions",
                type="primary"
            ):

                results, probabilities, predictions = (
                    analyze_transactions(input_data)
                )

                total = len(results)

                fraud = int(
                    (predictions == 1).sum()
                )

                genuine = total - fraud

                st.success("Analysis completed successfully!")

                # -------- KPI CARDS --------

                st.markdown("### Transaction Summary")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Total Transactions",
                    f"{total:,}"
                )

                c2.metric(
                    "Fraud Detected",
                    f"{fraud:,}"
                )

                c3.metric(
                    "Genuine Transactions",
                    f"{genuine:,}"
                )

                c4.metric(
                    "Fraud Percentage",
                    f"{(fraud / total * 100):.3f}%"
                    if total else "0%"
                )

                # -------- RISK DISTRIBUTION --------

                st.markdown("### Risk Distribution")

                risk_counts = (
                    results["Risk_Level"]
                    .value_counts()
                    .reindex(
                        ["Low", "Medium", "High"],
                        fill_value=0
                    )
                )

                st.bar_chart(risk_counts)

                # -------- FRAUD PROBABILITY --------

                st.markdown("### Fraud Probability Distribution")

                fig, ax = plt.subplots(figsize=(9, 4))

                ax.hist(
                    results["Fraud_Probability"],
                    bins=30
                )

                ax.set_xlabel("Fraud Probability (%)")
                ax.set_ylabel("Number of Transactions")
                ax.set_title("Fraud Probability Distribution")

                st.pyplot(fig)
                plt.close(fig)

                # -------- PREDICTION TABLE --------

                st.markdown("### Prediction Results")

                display_columns = [
                    "Amount",
                    "Fraud_Probability",
                    "Prediction",
                    "Risk_Level",
                    "Alert"
                ]

                st.dataframe(
                    results[display_columns].head(100),
                    use_container_width=True
                )

                st.caption(
                    "Showing the first 100 rows. "
                    "The complete results are available in the CSV report."
                )

                # -------- MODEL EVALUATION --------

                if has_labels:

                    st.divider()

                    st.header("Model Performance Evaluation")

                    valid_mask = actual.isin([0, 1])

                    if valid_mask.any():

                        y_true = actual[valid_mask].astype(int)
                        y_pred = predictions[valid_mask.to_numpy()]
                        y_prob = probabilities[valid_mask.to_numpy()]

                        accuracy = accuracy_score(
                            y_true, y_pred
                        )

                        precision = precision_score(
                            y_true, y_pred,
                            zero_division=0
                        )

                        recall = recall_score(
                            y_true, y_pred,
                            zero_division=0
                        )

                        f1 = f1_score(
                            y_true, y_pred,
                            zero_division=0
                        )

                        pr_auc = average_precision_score(
                            y_true, y_prob
                        )

                        st.warning(
                            "These metrics compare predictions with the "
                            "Class column in the uploaded dataset. If this "
                            "dataset was also used during model training, "
                            "these are not independent test metrics."
                        )

                        m1, m2, m3, m4, m5 = st.columns(5)

                        m1.metric(
                            "Accuracy",
                            f"{accuracy * 100:.2f}%"
                        )

                        m2.metric(
                            "Precision",
                            f"{precision * 100:.2f}%"
                        )

                        m3.metric(
                            "Recall",
                            f"{recall * 100:.2f}%"
                        )

                        m4.metric(
                            "F1-Score",
                            f"{f1 * 100:.2f}%"
                        )

                        m5.metric(
                            "PR-AUC",
                            f"{pr_auc:.4f}"
                        )

                        # -------- CONFUSION MATRIX --------

                        st.markdown("### Confusion Matrix")

                        cm = confusion_matrix(
                            y_true,
                            y_pred,
                            labels=[0, 1]
                        )

                        fig, ax = plt.subplots(figsize=(6, 4))

                        image = ax.imshow(
                            cm,
                            interpolation="nearest"
                        )

                        ax.set_title("Confusion Matrix")
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("Actual Label")

                        ax.set_xticks([0, 1])
                        ax.set_yticks([0, 1])

                        ax.set_xticklabels(
                            ["Genuine", "Fraud"]
                        )

                        ax.set_yticklabels(
                            ["Genuine", "Fraud"]
                        )

                        for i in range(2):
                            for j in range(2):
                                ax.text(
                                    j, i,
                                    str(cm[i, j]),
                                    ha="center",
                                    va="center"
                                )

                        fig.colorbar(image, ax=ax)

                        st.pyplot(fig)
                        plt.close(fig)

                        tn, fp, fn, tp = cm.ravel()

                        st.markdown("### Confusion Matrix Details")

                        a, b, c, d = st.columns(4)

                        a.metric("True Negative", f"{tn:,}")
                        b.metric("False Positive", f"{fp:,}")
                        c.metric("False Negative", f"{fn:,}")
                        d.metric("True Positive", f"{tp:,}")

                    else:
                        st.warning(
                            "The Class column has no valid binary labels."
                        )

                else:

                    st.info(
                        "No Class column found. Actual model performance "
                        "metrics cannot be calculated without true labels."
                    )

                # -------- DOWNLOAD REPORT --------

                st.divider()

                st.markdown("### Download Report")

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Fraud Analysis Report",
                    data=csv,
                    file_name="creditguard_fraud_report.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(f"Error: {e}")


# ---------------- ABOUT ----------------

elif page == "About":

    st.header("About CreditGuard AI")

    st.write(
        """
        CreditGuard AI is a machine learning based
        credit card fraud detection project.

        It analyzes transaction features and predicts
        whether a transaction is potentially fraudulent.

        ### Technologies

        - Python
        - Pandas and NumPy
        - Scikit-learn
        - XGBoost
        - Streamlit

        ### Main Features

        - Batch transaction analysis
        - Fraud probability prediction
        - Risk classification
        - Fraud alerts
        - Model evaluation
        - Downloadable reports
        """
    )

    st.warning(
        "This is an academic prototype. Predictions are not "
        "bank authorization decisions. Do not upload real "
        "cardholder information or sensitive financial data."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "CreditGuard AI | Academic ML Project"
)
