# Feature engineering.
#
# Training and the API both use this same function, so the model always gets
# its features calculated the same way.

KM_10K_TO_HALF = 21.0975 - 10   # distance from the 10K mat to the halfway mat


def add_features(df):
    df = df.copy()

    # Pace in minutes per km for the first 10K, and from 10K to halfway
    pace_first_10k = df["split_10k"] / 10
    pace_10k_to_half = (df["split_half"] - df["split_10k"]) / KM_10K_TO_HALF

    # Positive means the runner was already slowing down before halfway,
    # which usually means a bigger slowdown in the second half too.
    df["slowdown"] = pace_10k_to_half - pace_first_10k

    return df
