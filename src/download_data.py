# Download the cleaned data file from our Google Cloud Storage bucket.
#
# train.py calls download_data() as its first step, but you can also run this
# file by itself to just get the data: python src/download_data.py

import os
import yaml
from google.cloud import storage


def download_data(bucket_name, blob_name, local_path):
    # The cleaned file is public (it has no names in it), so nobody needs a
    # Google Cloud login to download it.
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded gs://{bucket_name}/{blob_name} to {local_path}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    download_data(config["data"]["bucket_name"],
                  config["data"]["blob_name"],
                  config["data"]["local_path"])
