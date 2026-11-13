# Download the cleaned data file from our Google Cloud Storage bucket.
#
# train.py calls download_data() as its first step, but you can also run this
# file by itself to just get the data: python src/download_data.py

import os
import yaml
from google.cloud import storage

# The cleaned files in the bucket are named boston_clean_v1.csv, boston_clean_v2.csv, ...
FILE_PREFIX = "boston_clean_v"


def download_data(bucket_name, version):
    # Downloads one version of the cleaned data and returns where it was saved.
    file_name = f"{FILE_PREFIX}{version}.csv"
    local_path = f"data/clean/{file_name}"

    # The cleaned file is public (it has no names in it), so nobody needs a
    # Google Cloud login to download it.
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    os.makedirs("data/clean", exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded gs://{bucket_name}/{file_name} to {local_path}")
    return local_path


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    download_data(config["data"]["bucket_name"], config["data"]["version"])
