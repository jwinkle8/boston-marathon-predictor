# The API that serves the trained model.
#
# Run it locally from the main folder of the repo (after training):
#   uvicorn api:app --app-dir src --port 8080
#
# Then open http://localhost:8080/docs to try it out.

import json
import os
from typing import Literal, Optional

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import CHECKPOINT_KM, add_features

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load the models once when the API starts, not on every request.
# There is one model for each set of checkpoint times a runner might know.
feature_sets = config["features"]["numeric"]
categorical = config["features"]["categorical"]
models = {}
metrics = {}
for name in feature_sets:
    model_dir = os.path.join(config["output"]["model_dir"], name)
    models[name] = joblib.load(os.path.join(model_dir, config["output"]["model_file"]))
    with open(os.path.join(model_dir, config["output"]["metrics_file"])) as f:
        metrics[name] = json.load(f)

SPLITS = list(CHECKPOINT_KM)   # in race order: 5K, 10K, halfway

app = FastAPI(title="Boston Marathon Finish Time Predictor")


# What a request to /predict has to look like. All times are in minutes.
# The limits are close to the range of the training data. The model has never
# seen anything outside of it, so we reject those instead of guessing.
# The three splits are optional: leave out (or send null for) the ones the
# runner doesn't know. At least one is needed.
class Runner(BaseModel):
    age: int = Field(ge=18, le=85)
    gender: Literal["M", "F"]
    split_5k: Optional[float] = Field(default=None, ge=14, le=60, description="Time at the 5K mark, in minutes")
    split_10k: Optional[float] = Field(default=None, ge=29, le=120, description="Time at the 10K mark, in minutes")
    split_half: Optional[float] = Field(default=None, ge=63, le=300, description="Time at the halfway mark, in minutes")


def minutes_to_clock(minutes):
    # 221.5 -> "3:41:30"
    total_seconds = round(minutes * 60)
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}:{mins:02d}:{secs:02d}"


def choose_model(known_splits):
    # With all three splits we use the original model. Otherwise we use the
    # model for the furthest checkpoint the runner knows, because the later
    # the checkpoint, the better the prediction.
    if len(known_splits) == len(SPLITS):
        return "all_splits"
    return known_splits[-1].replace("split_", "") + "_only"


@app.get("/")
def home():
    return {"message": "Boston Marathon finish time API. Send a POST request to /predict."}


@app.post("/predict")
def predict(runner: Runner):
    known = {k: v for k, v in runner.model_dump().items() if v is not None}
    known_splits = [s for s in SPLITS if s in known]

    if not known_splits:
        raise HTTPException(status_code=400,
                            detail="Enter at least one checkpoint time (5K, 10K or halfway).")

    # You can't reach 10K before 5K, so this means something was typed in wrong
    times = [known[s] for s in known_splits]
    if any(earlier >= later for earlier, later in zip(times, times[1:])):
        raise HTTPException(status_code=400,
                            detail="The splits have to go up: 5K time < 10K time < halfway time.")

    name = choose_model(known_splits)
    row = add_features(pd.DataFrame([known]))
    row = row[feature_sets[name] + categorical]
    prediction = float(models[name].predict(row)[0])

    return {
        "predicted_finish_minutes": round(prediction, 1),
        "predicted_finish_time": minutes_to_clock(prediction),
        # average error on the test set, so the app can show "give or take X minutes"
        "typical_error_minutes": metrics[name]["test_mae"],
        # which model made the prediction, and which of the splits it actually used
        "model_used": name,
        "splits_used": [s for s in known_splits if s in feature_sets[name]],
    }
