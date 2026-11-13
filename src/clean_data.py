# Clean the raw Boston Marathon results (2015-2017) from Kaggle.
#
# Run this when the raw data or the cleaning steps change. Every run saves a
# new version of the cleaned file (v1, v2, ...), so an older version never gets
# overwritten. The cleaned file is what we upload to Google Cloud Storage, and
# the training pipeline downloads it from there.
#
# Before running, download the three files from Kaggle and put them in data/raw/:
#   marathon_results_2015.csv, marathon_results_2016.csv, marathon_results_2017.csv
#
# Run from the main folder of the repo: python src/clean_data.py

import os

import pandas as pd
import yaml
from google.cloud import storage

from download_data import FILE_PREFIX

YEARS = [2015, 2016, 2017]

# The only columns we need. We pick them by name because the three files
# don't have exactly the same columns.
# Everything else gets dropped on purpose:
#   - Bib, City, State, Country: personal info we don't need
#   - Pace, Proj Time, Overall, Gender, Division: these are calculated from the
#     finish time, so keeping them would give away the answer (leakage)
#   - 25K to 40K splits: we want to predict from the halfway point
KEEP = ["Name", "Age", "M/F", "5K", "10K", "Half", "Official Time"]


def time_to_minutes(column):
    # Times are text like "1:45:30". Missing splits are "-", and errors="coerce"
    # turns those into NaN instead of crashing.
    return pd.to_timedelta(column, errors="coerce").dt.total_seconds() / 60


def next_version(bucket_name):
    # Looks at the cleaned files that are already in the bucket and returns the
    # next version number. If the bucket has v1 and v2, this returns 3.
    client = storage.Client.create_anonymous_client()
    versions = []
    for blob in client.list_blobs(bucket_name, prefix=FILE_PREFIX):
        # file names look like boston_clean_v2.csv
        number = blob.name.replace(FILE_PREFIX, "").replace(".csv", "")
        versions.append(int(number))
    return max(versions, default=0) + 1


with open("config.yaml") as f:
    config = yaml.safe_load(f)


# ----- Load the three years and stack them -----
frames = []
for year in YEARS:
    df = pd.read_csv(f"data/raw/marathon_results_{year}.csv")
    df = df[KEEP].copy()
    df["year"] = year
    frames.append(df)

df = pd.concat(frames, ignore_index=True)
print("Rows in raw data:", len(df))

# ----- Rename columns and convert all times to minutes -----
df = df.rename(columns={"Age": "age", "M/F": "gender"})
df["split_5k"] = time_to_minutes(df["5K"])
df["split_10k"] = time_to_minutes(df["10K"])
df["split_half"] = time_to_minutes(df["Half"])
df["finish_time"] = time_to_minutes(df["Official Time"])

# ----- Drop rows with missing splits (the timing mat didn't record them) -----
before = len(df)
df = df.dropna(subset=["split_5k", "split_10k", "split_half", "finish_time"])
print("Dropped for missing splits:", before - len(df))

# ----- Drop rows where the splits are out of order (timing errors) -----
before = len(df)
in_order = (df["split_5k"] < df["split_10k"]) & (df["split_10k"] < df["split_half"]) & (df["split_half"] < df["finish_time"])
df = df[in_order]
print("Dropped for splits out of order:", before - len(df))

# ----- Give each runner an ID, then drop the name -----
# A lot of people ran Boston more than once in these three years. We want all of
# one person's races to end up on the same side of the train/test split, so we
# need a way to tell who is who. Name + gender + birth year is close enough.
# The ID is just a number, so the cleaned file has no names in it.
birth_year = df["year"] - df["age"]
runner_key = df["Name"] + "_" + df["gender"] + "_" + birth_year.astype(str)
df["runner_id"] = pd.factorize(runner_key)[0]

df = df[["runner_id", "year", "age", "gender", "split_5k", "split_10k", "split_half", "finish_time"]]
df = df.round(2)

print("Rows in cleaned data:", len(df))
print("Different runners:", df["runner_id"].nunique())
print(df.describe().round(1))

# ----- Save as a new version -----
version = next_version(config["data"]["bucket_name"])
output_path = f"data/clean/{FILE_PREFIX}{version}.csv"

os.makedirs("data/clean", exist_ok=True)
df.to_csv(output_path, index=False)
print("Saved", output_path)
print(f"To use it, upload it to the bucket and set data.version to {version} in config.yaml")
