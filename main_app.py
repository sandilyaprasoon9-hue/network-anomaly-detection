import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from anomaly_model import train_and_save_model, load_model
from data_utils import validate_data, detect_anomalies

st.set_page_config(page_title="Network Anomaly Detection", layout="wide")

# ✅ FORCE UI TO SHOW
st.title("🚨 Network Anomaly Detection Dashboard")
st.write("✅ App loaded successfully")
st.info("👆 Upload a CSV file to begin")

uploaded_file = st.file_uploader("📂 Upload Network CSV File", type=["csv"])

# 👇 THIS PART WAS THE PROBLEM
if uploaded_file is None:
    st.warning("Please upload a file to continue")
else:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df)

    if not validate_data(df):
        st.error("CSV must contain: packets, latency, errors")
    else:
        # Train model only once
        if not os.path.exists("model/isolation_forest.pkl"):
            train_and_save_model(df[["packets", "latency", "errors"]])

        model = load_model()

        df = detect_anomalies(model, df)

        st.subheader("🔍 Processed Data")
        st.dataframe(df)

        anomaly_count = (df["anomaly"] == -1).sum()

        col1, col2 = st.columns(2)
        col1.metric("Total Records", len(df))
        col2.metric("Anomalies Found", anomaly_count)

        feature = st.selectbox("📌 Select Feature", ["packets", "latency"])

        fig, ax = plt.subplots()

        normal = df[df["anomaly"] == 1]
        anomaly = df[df["anomaly"] == -1]

        ax.plot(normal[feature].values, label="Normal")
        ax.scatter(anomaly.index, anomaly[feature], color="red", label="Anomaly")

        ax.set_title(f"{feature} Trend with Anomalies")
        ax.legend()

        st.pyplot(fig)

        st.subheader("⚠️ Alerts")

        if anomaly_count > 0:
            st.error(f"🚨 {anomaly_count} anomalies detected!")
        else:
            st.success("✅ No anomalies detected")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")
