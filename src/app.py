import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from metaflow import Flow
import logging
import re, pickle
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress all warnings
warnings.filterwarnings("ignore")

app = FastAPI()


class PredictionRequest(BaseModel):
    title: str
    loc_string: Optional[str] = None
    loc: Optional[str] = None
    features: Optional[List[str]] = None  # Add features field
    description: Optional[str] = None  # Add description field
    type: Optional[str] = None
    subtype: Optional[str] = None
    selltype: Optional[str] = None
    desc: Optional[str] = None  # Add description field
    id: Optional[int] = 0


# Define the path for saving predictions
CSV_FILE_PATH = "solution.csv"

# Load MLFlow model
mlflow.set_tracking_uri('https://mlflow-539716308541.us-west2.run.app/')
MODEL_URI = "models:/metaflow-price-model-gcp/None"
# MODEL_URI = "mlflow-artifacts:/3/08c80d67056a4862ab28f4e0fed1e310/artifacts/metaflow_train"
model = None

# Load model from MLFlow
def load_model():
    global model
    if model is None:
        model = mlflow.pyfunc.load_model(MODEL_URI)

# Load data from the latest Metaflow run of 'PriceTrainFlow'
def load_data_from_config():
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions("metaflow-price-model-gcp", stages=["None"])[0]
    run_id = latest_version.run_id
    local_dir = mlflow.artifacts.download_artifacts(
        artifact_path="metaflow_train/test_config.pkl",
        run_id=run_id
    )
    with open(local_dir, 'rb') as f:
        data = pickle.load(f)
    return data

# Reformat incoming request data into a DataFrame for prediction
def reformat_request_data(data, test_config):
    inp = [data.dict()]
    for i in range(len(inp)):
        try:
            if type(inp[i]['price']) is str:
                t = ''
                for a in inp[i]['price']:
                    if a in '0123456789':
                        t += a
                inp[i]['price'] = int(t)
        except KeyError:
            pass
        try:
            for a in inp[i]['features']:
                s = a.split(' ')
                if s[1] == 'm2':
                    inp[i]['area'] = int(s[0])
                elif s[1] == 'hab.':
                    inp[i]['bed'] = int(s[0])
                elif s[1] == 'ba\u00f1o':
                    inp[i]['bath'] = int(s[0])
            del inp[i]['features']
        except KeyError:
            pass
    df = pd.DataFrame(inp)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    # # Type hint for Pylint
    # x_test: pd.DataFrame  # Explicitly declaring that x_test is a DataFrame
    df = df[test_config["columns_orig"]]
    # Apply preprocessing based on data from Metaflow (e.g., column names, scaling)
    df = pd.get_dummies(df, columns=["loc_string", "loc", "type", "subtype", "selltype"])
    all_columns = set(test_config['all_colummns'])
    missing_columns = all_columns - set(df.columns)
    for col in missing_columns:
        df[col] = 0

    # Fill missing values and scale numeric data
    df["bath"] = df["bath"].fillna(1)
    df["bed"] = df["bed"].fillna(1)
    df["area"] = df["area"].fillna(test_config["area"])
    
    # Standardize numeric data
    numeric_columns = test_config['numeric_columns']
    df[numeric_columns] = test_config["scaler"].transform(df[numeric_columns])
    logger.info(df)
    return df

# Save predictions to CSV
def save_predictions_to_csv(input_data, prediction):
    df = input_data.copy()
    df['predicted_price'] = prediction
    df.index.name = 'id'
    df.to_csv(CSV_FILE_PATH, mode='a', header=not Path(CSV_FILE_PATH).is_file(), index=True)
    return CSV_FILE_PATH

@app.get('/')
def main():
    return {'message': 'This is a model for price prediction'}

@app.post("/predict")
def predict(data: PredictionRequest):
    load_model()
    try:
        # Load data from mlflow to use for preprocessing
        config = load_data_from_config()
        # Reformat the incoming data into a DataFrame for prediction
        input_data = reformat_request_data(data, config)
        # Make prediction using the loaded MLFlow model
        prediction = model.predict(input_data)
        logger.info(prediction)
        # Save prediction to CSV
        file_path = save_predictions_to_csv(input_data, prediction[0])
        
        # Return prediction as a response
        return {"predicted_price": prediction[0], "file_path": file_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For running the app locally: use 'uvicorn app:app --reload'
#sample request
# curl -X 'POST' \
#   'http://localhost:8000/predict' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "title": "Piso Carrer de llull. Piso con 4 habitaciones con ascensor y calefacción",
#   "loc_string": "Barcelona - El Parc i la Llacuna del Poblenou",
#   "loc": "None",
#   "features": ["87 m2", "4 hab.", "1 baño"],
#   "description": "Contactar con Camila 7. 3. La Casa Agency Estudio Miraflores tiene el placer de presentarles esta es...",
#   "type": "FLAT",
#   "subtype": "FLAT",
#   "selltype": "SECOND_HAND",
#   "desc": "Contactar con Camila 7. 3.La Casa Agency Estudio Miraflores tiene",
#   "id": 0
# }'