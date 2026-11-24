# Train the model that predicts a runner's Boston Marathon finish time.
#
# Steps:
#   1. download the cleaned data from Google Cloud Storage
#   2. add features
#   3. split into train / validation / test
#   4. train the models (one for each set of features in config.yaml)
#   5. evaluate them
#   6. log everything to MLflow and save the models for the API
#
# All the settings come from config.yaml.
# Run from the main folder of the repo: python src/train.py

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from download_data import download_data
from features import add_features, even_pace_finish, latest_split

with open("config.yaml") as f:
    config = yaml.safe_load(f)

feature_sets = config["features"]["numeric"]     # model name -> its numeric features
categorical = config["features"]["categorical"]
target = config["target"]
seed = config["split"]["random_seed"]


# ----- 1. Get the data from the cloud -----
data_path = download_data(config["data"]["bucket_name"], config["data"]["version"])
df = pd.read_csv(data_path)
print("Rows:", len(df))


# ----- 2. Feature engineering -----
df = add_features(df)


# ----- 3. Train / validation / test split -----
# About 1 in 7 runners is in the data more than once (they ran in more than one
# year). If the same person is in both train and test, the scores look better
# than they really are. So we split the runner IDs, not the rows.
test_size = config["split"]["test_size"]
val_size = config["split"]["val_size"]

runner_ids = df["runner_id"].unique()
train_ids, test_ids = train_test_split(runner_ids, test_size=test_size, random_state=seed)
train_ids, val_ids = train_test_split(train_ids, test_size=val_size / (1 - test_size), random_state=seed)

train = df[df["runner_id"].isin(train_ids)]
val = df[df["runner_id"].isin(val_ids)]
test = df[df["runner_id"].isin(test_ids)]
print("Train:", len(train), "| Validation:", len(val), "| Test:", len(test))

model_type = config["model"]["type"]
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# Every model is trained and tested on the same runners, so their scores can be
# compared directly. The only thing that changes is which features they get.
for name, numeric in feature_sets.items():
    print(f"\n===== Model: {name} ({', '.join(numeric)}) =====")
    y_train, y_val, y_test = train[target], val[target], test[target]
    X_train = train[numeric + categorical]
    X_val = val[numeric + categorical]
    X_test = test[numeric + categorical]

    # ----- 4. Build and train the model -----
    # Preprocessing and the model go in one sklearn Pipeline, so the API can load a
    # single file and get the exact same preprocessing that was used in training.
    preprocess = ColumnTransformer([
        ("gender", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numbers", "passthrough", numeric),
    ])

    if model_type == "random_forest":
        regressor = RandomForestRegressor(n_estimators=config["model"]["n_estimators"],
                                          max_depth=config["model"]["max_depth"],
                                          random_state=seed)
    elif model_type == "linear_regression":
        regressor = LinearRegression()
    else:
        raise ValueError("model type in config.yaml must be random_forest or linear_regression")

    model = Pipeline([("preprocess", preprocess), ("regressor", regressor)])
    model.fit(X_train, y_train)

    # ----- 5. Evaluate -----
    val_mae, val_rmse, val_r2 = evaluate(y_val, model.predict(X_val))
    test_pred = model.predict(X_test)
    test_mae, test_rmse, test_r2 = evaluate(y_test, test_pred)

    # Baseline to compare against: the runner holds the pace from their latest
    # checkpoint all the way to the finish (for halfway: double the half split)
    baseline_split = latest_split(numeric)
    baseline_mae, baseline_rmse, baseline_r2 = evaluate(y_test, even_pace_finish(test, baseline_split))

    print(f"Validation: MAE = {val_mae:.2f} min, RMSE = {val_rmse:.2f}, R2 = {val_r2:.3f}")
    print(f"Test:       MAE = {test_mae:.2f} min, RMSE = {test_rmse:.2f}, R2 = {test_r2:.3f}")
    print(f"Baseline (even pace from {baseline_split}) on test: MAE = {baseline_mae:.2f} min")

    # Plot predicted vs actual finish time on the test set
    model_dir = os.path.join(config["output"]["model_dir"], name)
    os.makedirs(model_dir, exist_ok=True)
    plot_path = os.path.join(model_dir, config["output"]["plot_file"])
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, test_pred, s=3, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
    plt.xlabel("Actual finish time (minutes)")
    plt.ylabel("Predicted finish time (minutes)")
    plt.title(f"Test set, {name}: predicted vs actual (MAE = {test_mae:.1f} min)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    # ----- 6. Save the model and log the run to MLflow -----
    model_path = os.path.join(model_dir, config["output"]["model_file"])
    joblib.dump(model, model_path)

    # The API also reports the typical error, so save the metrics next to the model
    metrics = {
        "val_mae": round(val_mae, 2), "val_rmse": round(val_rmse, 2), "val_r2": round(val_r2, 4),
        "test_mae": round(test_mae, 2), "test_rmse": round(test_rmse, 2), "test_r2": round(test_r2, 4),
        "baseline_test_mae": round(baseline_mae, 2),
    }
    with open(os.path.join(model_dir, config["output"]["metrics_file"]), "w") as f:
        json.dump(metrics, f, indent=2)

    with mlflow.start_run(run_name=f"{model_type}_{name}"):
        mlflow.log_params(config["model"])
        mlflow.log_params(config["split"])
        mlflow.log_param("model_name", name)
        mlflow.log_param("features", numeric + categorical)
        mlflow.log_param("baseline_split", baseline_split)
        mlflow.log_param("data_version", config["data"]["version"])
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("val_rows", len(val))
        mlflow.log_param("test_rows", len(test))

        mlflow.log_metrics(metrics)

        mlflow.log_artifact("config.yaml")
        mlflow.log_artifact(plot_path)
        # MLflow's default save format won't save tree models unless we say we trust that type
        mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(5),
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])

    print("Saved model to", model_path)

print("\nLogged", len(feature_sets), "runs to MLflow experiment:", config["mlflow"]["experiment_name"])
