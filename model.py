import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def train_and_save_model():
    np.random.seed(42)
    n = 500

    sleep = np.random.uniform(3, 9, n)
    screen_time = np.random.uniform(1, 12, n)
    activity = np.random.uniform(0, 10, n)

    # Stress logic: less sleep + more screen + less activity = higher stress
    stress_score = (
        (9 - sleep) * 0.4 +
        screen_time * 0.35 +
        (10 - activity) * 0.25
    )

    labels = pd.cut(stress_score, bins=3, labels=["Low", "Medium", "High"])

    df = pd.DataFrame({
        "sleep_hours": sleep,
        "screen_time": screen_time,
        "activity_level": activity,
        "stress_level": labels
    })

    X = df[["sleep_hours", "screen_time", "activity_level"]]
    y = df["stress_level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved.")

if __name__ == "__main__":
    train_and_save_model()
