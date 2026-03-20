import streamlit as st
import pickle
import numpy as np
import os
from model import train_and_save_model

# Train model if not already saved
if not os.path.exists("model.pkl"):
    train_and_save_model()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# --- UI ---
st.set_page_config(page_title="Stress Predictor", page_icon="🧠")
st.title("🧠 AI Stress Prediction System")
st.markdown("Enter your daily habits below to predict your stress level.")

st.divider()

sleep = st.slider("😴 Sleep Hours (last night)", 2.0, 10.0, 7.0, step=0.5)
screen = st.slider("📱 Screen Time (hours/day)", 0.0, 14.0, 6.0, step=0.5)
activity = st.slider("🏃 Physical Activity Level (0 = none, 10 = very active)", 0.0, 10.0, 5.0, step=0.5)

if st.button("Predict Stress Level"):
    input_data = np.array([[sleep, screen, activity]])
    prediction = model.predict(input_data)[0]

    color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    st.markdown(f"## Stress Level: {color.get(prediction, '')} **{prediction}**")

    if prediction == "High":
        st.warning("Consider reducing screen time and improving sleep.")
    elif prediction == "Medium":
        st.info("You're doing okay — small improvements can help.")
    else:
        st.success("Great job! Keep up your healthy habits.")

st.divider()
st.caption("Built with Python, Scikit-learn & Streamlit")
```

---

### `requirements.txt`
```
streamlit
scikit-learn
pandas
numpy
