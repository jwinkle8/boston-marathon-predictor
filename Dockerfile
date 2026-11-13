# One image for the whole project: it trains the model and serves the API.
FROM python:3.11-slim

WORKDIR /app

# Install the packages first. Docker caches this step, so rebuilding after a
# code change is fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/

# MLflow tries to save the git commit with each run, but there's no git inside
# this image. This stops it from printing a long warning about that.
ENV GIT_PYTHON_REFRESH=quiet

# Training entry point.
# Downloads the data from our cloud bucket and trains the model while the image
# is being built, so the finished image always has a trained model inside.
# To train again in a container: docker run marathon-api python src/train.py
RUN python src/train.py

# Serving entry point.
# Cloud Run tells the container which port to listen on with the PORT variable.
# 8080 is the default when running locally.
ENV PORT=8080
EXPOSE 8080
CMD uvicorn api:app --app-dir src --host 0.0.0.0 --port $PORT
