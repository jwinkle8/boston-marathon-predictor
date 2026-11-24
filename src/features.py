# Feature engineering.
#
# Training and the API both use this same function, so the model always gets
# its features calculated the same way.

# Distance of each checkpoint mat from the start line, in km
CHECKPOINT_KM = {"split_5k": 5, "split_10k": 10, "split_half": 21.0975}
MARATHON_KM = 42.195

KM_10K_TO_HALF = CHECKPOINT_KM["split_half"] - CHECKPOINT_KM["split_10k"]


def add_features(df):
    df = df.copy()

    # slowdown needs both the 10K and the halfway time. A runner who only gave
    # us one checkpoint time doesn't get it (and their model doesn't use it).
    if "split_10k" in df.columns and "split_half" in df.columns:
        # Pace in minutes per km for the first 10K, and from 10K to halfway
        pace_first_10k = df["split_10k"] / CHECKPOINT_KM["split_10k"]
        pace_10k_to_half = (df["split_half"] - df["split_10k"]) / KM_10K_TO_HALF

        # Positive means the runner was already slowing down before halfway,
        # which usually means a bigger slowdown in the second half too.
        df["slowdown"] = pace_10k_to_half - pace_first_10k

    return df


def latest_split(columns):
    # The furthest checkpoint in a list of column names, e.g.
    # ["age", "split_5k", "split_10k"] -> "split_10k"
    splits = [c for c in columns if c in CHECKPOINT_KM]
    return max(splits, key=CHECKPOINT_KM.get)


def even_pace_finish(df, split):
    # The simple guess to compare the model against: the runner keeps the same
    # pace they had up to this checkpoint. For the halfway split this is just
    # "double your halfway time".
    return df[split] * MARATHON_KM / CHECKPOINT_KM[split]
