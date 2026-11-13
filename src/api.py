# The API that serves the trained model.
#
# Run it locally from the main folder of the repo (after training):
#   uvicorn api:app --app-dir src --port 8080
#
# Then open http://localhost:8080/docs to try it out.

import json
from typing import Literal

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import add_features

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load the model once when the API starts, not on every request
model = joblib.load(config["output"]["model_path"])
with open(config["output"]["metrics_path"]) as f:
    metrics = json.load(f)

app = FastAPI(title="Boston Marathon Finish Time Predictor")


# What a request to /predict has to look like. All times are in minutes.
# The limits are close to the range of the training data. The model has never
# seen anything outside of it, so we reject those instead of guessing.
class Runner(BaseModel):
    age: int = Field(ge=18, le=85)
    gender: Literal["M", "F"]
    split_5k: float = Field(ge=14, le=60, description="Time at the 5K mark, in minutes")
    split_10k: float = Field(ge=29, le=120, description="Time at the 10K mark, in minutes")
    split_half: float = Field(ge=63, le=300, description="Time at the halfway mark, in minutes")


def minutes_to_clock(minutes):
    # 221.5 -> "3:41:30"
    total_seconds = round(minutes * 60)
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}:{mins:02d}:{secs:02d}"


@app.get("/")
def home():
    return {"message": "Boston Marathon finish time API. Send a POST request to /predict."}


@app.post("/predict")
def predict(runner: Runner):
    # You can't reach 10K before 5K, so this means something was typed in wrong
    if not (runner.split_5k < runner.split_10k < runner.split_half):
        raise HTTPException(status_code=400,
                            detail="The splits have to go up: 5K time < 10K time < halfway time.")

    row = pd.DataFrame([runner.model_dump()])
    row = add_features(row)
    prediction = float(model.predict(row)[0])

    return {
        "predicted_finish_minutes": round(prediction, 1),
        "predicted_finish_time": minutes_to_clock(prediction),
        # average error on the test set, so the app can show "give or take X minutes"
        "typical_error_minutes": metrics["test_mae"],
    }
