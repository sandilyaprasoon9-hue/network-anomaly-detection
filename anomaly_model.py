from sklearn.ensemble import IsolationForest
import pickle
import os

MODEL_PATH = "model/isolation_forest.pkl"

def train_and_save_model(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data)

    os.makedirs("model", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
