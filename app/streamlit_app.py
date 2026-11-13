# Streamlit front-end for the marathon finish time predictor.
#
# It doesn't load the model itself. It sends the runner's info to our deployed
# API and shows the prediction that comes back.
#
# Run locally: streamlit run app/streamlit_app.py
# To point it at an API running on your own computer:
#   API_URL=http://localhost:8080 streamlit run app/streamlit_app.py

import os

import requests
import streamlit as st

# URL of the deployed API on Cloud Run
API_URL = os.environ.get("API_URL", "https://REPLACE-WITH-CLOUD-RUN-URL")


def to_minutes(hours, minutes, seconds):
    return hours * 60 + minutes + seconds / 60


def to_clock(minutes):
    # 107.5 -> "1:47:30"
    total_seconds = round(minutes * 60)
    return f"{total_seconds // 3600}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"


st.title("Boston Marathon Finish Time Predictor")
st.write(
    "Enter your age, gender and your times at the first three checkpoints of the "
    "race, and the model will predict your finish time."
)
st.caption(
    "The checkpoint times are your times *during the marathon*. For example, the halfway "
    "time is when you crossed the 13.1 mile mark, not your best half marathon race time."
)

# ----- Inputs -----
col1, col2 = st.columns(2)
age = col1.number_input("Age", min_value=18, max_value=85, value=35)
gender = col2.radio("Gender", ["Female", "Male"], horizontal=True)

st.subheader("Your checkpoint times")

col1, col2 = st.columns(2)
min_5k = col1.number_input("5K: minutes", min_value=0, max_value=60, value=25)
sec_5k = col2.number_input("5K: seconds", min_value=0, max_value=59, value=30)

col1, col2 = st.columns(2)
min_10k = col1.number_input("10K: minutes", min_value=0, max_value=120, value=51)
sec_10k = col2.number_input("10K: seconds", min_value=0, max_value=59, value=0)

col1, col2, col3 = st.columns(3)
hr_half = col1.number_input("Halfway: hours", min_value=1, max_value=5, value=1)
min_half = col2.number_input("Halfway: minutes", min_value=0, max_value=59, value=48)
sec_half = col3.number_input("Halfway: seconds", min_value=0, max_value=59, value=0)

# ----- Send the inputs to the API -----
if st.button("Predict my finish time", type="primary"):
    runner = {
        "age": age,
        "gender": "F" if gender == "Female" else "M",
        "split_5k": to_minutes(0, min_5k, sec_5k),
        "split_10k": to_minutes(0, min_10k, sec_10k),
        "split_half": to_minutes(hr_half, min_half, sec_half),
    }

    try:
        with st.spinner("Asking the model..."):
            # long timeout because the API can take a while to wake up if nobody has used it recently
            response = requests.post(API_URL + "/predict", json=runner, timeout=60)
    except requests.exceptions.RequestException:
        st.error("Couldn't reach the API. It might still be waking up, so try again in a few seconds.")
        st.stop()

    if response.status_code == 200:
        result = response.json()
        finish = result["predicted_finish_minutes"]
        error = result["typical_error_minutes"]
        first_half = runner["split_half"]
        second_half = finish - first_half

        st.metric("Predicted finish time", result["predicted_finish_time"])
        st.write(
            f"Give or take about **{error:.0f} minutes**. That's how far off the model "
            "was on average for runners it had never seen."
        )

        col1, col2 = st.columns(2)
        col1.metric("Your first half", to_clock(first_half))
        col2.metric("Predicted second half", to_clock(second_half),
                    delta=f"{second_half - first_half:+.1f} min vs first half", delta_color="inverse")

    elif response.status_code == 400:
        # our API's own check, like splits that are out of order
        st.error(response.json()["detail"])

    elif response.status_code == 422:
        # a value was outside the range the API accepts
        for problem in response.json()["detail"]:
            st.error(f"{problem['loc'][-1]}: {problem['msg']}")

    else:
        st.error(f"Something went wrong (error {response.status_code}).")

# ----- Notes about the model -----
with st.expander("About this prediction"):
    st.write(
        "The model was trained on about 79,000 finishers of the 2015, 2016 and 2017 "
        "Boston Marathons. A few things to keep in mind:"
    )
    st.write(
        "- Most Boston runners had to qualify, so they're faster and more experienced "
        "than the average marathoner. If this is your first marathon, expect to slow "
        "down more than the model says.\n"
        "- Boston's course is downhill early and has big hills late, so the prediction "
        "fits Boston better than a flat course.\n"
        "- The results only record gender as male or female, so those are the only "
        "options here.\n"
        "- It's an estimate for fun, not a guarantee. Weather, fueling and how you feel "
        "on the day matter a lot."
    )
