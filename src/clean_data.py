# Clean the raw Boston Marathon results (2015-2017) from Kaggle.
#
# This only needs to be run once. The cleaned file it makes is what we upload
# to Google Cloud Storage, and the training pipeline downloads it from there.
#
# Before running, download the three files from Kaggle and put them in data/raw/:
#   marathon_results_2015.csv, marathon_results_2016.csv, marathon_results_2017.csv
#
# Run from the main folder of the repo: python src/clean_data.py

import os

import pandas as pd

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

os.makedirs("data/clean", exist_ok=True)
df.to_csv("data/clean/boston_clean_v1.csv", index=False)
print("Saved data/clean/boston_clean_v1.csv")
