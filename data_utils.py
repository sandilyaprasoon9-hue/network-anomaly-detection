def validate_data(df):
    required_cols = ["packets", "latency", "errors"]
    return all(col in df.columns for col in required_cols)

def detect_anomalies(model, df):
    df["anomaly"] = model.predict(df[["packets", "latency", "errors"]])
    return df
