import joblib
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "invoice_flagging" / "models" / "scaler.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)

def load_scaler(scaler_path: str = SCALER_PATH):
    return joblib.load(scaler_path)

def predict_invoice_flag(input_data):
    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)
    input_df = input_df[FEATURES]  # ensure correct column order

    scaled = scaler.transform(input_df)
    input_df['Predicted_flag'] = model.predict(scaled)
    return input_df

if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [100, 50, 20, 5],
        "invoice_dollars": [18500, 9000, 3000, 200],
        "Freight": [1200, 600, 150, 20],
        "total_item_quantity": [95, 48, 22, 5],
        "total_item_dollars": [18000, 8800, 2950, 195]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)