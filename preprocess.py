import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pyttsx3

# -------------------------------
# TEXT TO SPEECH SETUP
# -------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -------------------------------
# 1. DATA GENERATION
# -------------------------------
def generate_data(n=2000):
    np.random.seed(42)

    data = {
        'traffic_density': np.random.choice([0, 1, 2], n),  # Low, Medium, High
        'avg_speed': np.random.randint(15, 70, n),
        'speed_limit': np.random.choice([25, 30, 40, 50], n),
        'road_width': np.random.choice([2, 3, 4], n),
        'visibility': np.random.randint(40, 100, n),
        'is_school_hour': np.random.choice([0, 1], n),
        'has_signals': np.random.choice([0, 1], n),
        'weather_condition': np.random.choice([0, 1, 2], n)
    }

    df = pd.DataFrame(data)
    overspeed = df['avg_speed'] - df['speed_limit']

    score = (
        df['traffic_density'] * 35 +
        overspeed.clip(lower=0) * 1.3 +
        (100 - df['visibility']) * 0.5 +
        df['is_school_hour'] * 30 +
        df['weather_condition'] * 15 -
        df['road_width'] * 10 -
        df['has_signals'] * 15
    )

    df['risk_level'] = pd.cut(
        score,
        bins=[-float('inf'), 60, 110, float('inf')],
        labels=['Low', 'Medium', 'High']
    )

    return df

# -------------------------------
# 2. MODEL TRAINING
# -------------------------------
df = generate_data()
X = df.drop('risk_level', axis=1)
y = df['risk_level']

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# -------------------------------
# 3. USER INPUT (TEXT ONLY)
# -------------------------------
print("\n" + "="*40)
print(" SMART SCHOOL ZONE RISK ANALYZER ")
print("="*40)

try:
    density_map = {"low": 0, "medium": 1, "high": 2}

    density = density_map[input("Traffic Density (Low / Medium / High): ").lower()]
    speed = int(input("Current Average Speed: "))
    speed_limit = int(input("Speed Limit: "))
    road_width = int(input("Road Width (lanes): "))
    visibility = int(input("Visibility %: "))
    school_hour = int(input("Is School Hour? (1=Yes, 0=No): "))
    signals = int(input("Signals Present? (1=Yes, 0=No): "))
    weather = int(input("Weather (0=Clear, 1=Rain, 2=Storm): "))

    user_data = pd.DataFrame([[
        density, speed, speed_limit, road_width,
        visibility, school_hour, signals, weather
    ]], columns=X.columns)

    # -------------------------------
    # 4. PREDICTION
    # -------------------------------
    prediction = model.predict(user_data)[0]
    confidence = max(model.predict_proba(user_data)[0]) * 100

    # -------------------------------
    # 5. SPEED REDUCTION LOGIC
    # -------------------------------
    recommended_speed = speed
    reason = ""

    if prediction == "High":
        recommended_speed = min(speed_limit, int(speed * 0.7))
        reason = "High risk due to current conditions."
    elif prediction == "Medium":
        recommended_speed = int(speed * 0.85)
        reason = "Moderate risk detected."
    else:
        reason = "Risk level is low. Current speed is acceptable."

    # -------------------------------
    # 6. TEXT OUTPUT
    # -------------------------------
    print("\n" + "-"*40)
    print(f"RISK LEVEL            : {prediction.upper()}")
    print(f"PREDICTION CONFIDENCE : {confidence:.2f}%")
    print(f"CURRENT SPEED         : {speed}")
    print(f"RECOMMENDED SPEED     : {recommended_speed}")
    print(f"ANALYSIS              : {reason}")
    print("-"*40)

    # -------------------------------
    # 7. VOICE OUTPUT
    # -------------------------------
    speak(
        f"The predicted risk level is {prediction}. "
        f"Confidence is {confidence:.1f} percent. "
        f"The recommended speed is {recommended_speed}."
    )

except Exception as e:
    print("❌ Input Error:", e)
    speak("There was an error in the input. Please try again.")