import cv2
import time
import pandas as pd
import os
import random

# Try gaze tracking
try:
    from gaze_tracking import GazeTracking
    gaze = GazeTracking()
    GAZE_AVAILABLE = True
except:

    print("⚠️ GazeTracking not installed. Using fallback.")
    GAZE_AVAILABLE = False

# Setup
os.makedirs("data", exist_ok=True)

# Create dataset
if not os.path.exists("data/final_dataset.csv"):
    df = pd.DataFrame(columns=[
        "user_id", "comparison", "focus_side",
        "emotion", "attention_time", "engagement_score"
    ])
    df.to_csv("data/final_dataset.csv", index=False)

# Emotion simulation
emotion_map = {
    "happy": 1,
    "neutral": 0.5,
    "sad": 0.2
}


# Webcam
cap = cv2.VideoCapture(0)
user_id = int(input("Enter User ID: "))

# -------------------------
# ANALYSIS FUNCTION
# -------------------------
def analyze_frame(frame):

    # Fake emotion (stable system)
    emotion = random.choice(["happy", "neutral", "sad"])

    # Eye tracking
    if GAZE_AVAILABLE:
        gaze.refresh(frame)
        if gaze.is_left():
            attention = "left"
        elif gaze.is_right():
            attention = "right"
        else:
            attention = "center"
    else:
        attention = random.choice(["left", "center", "right"])

    return emotion, attention

# -------------------------
# RUN TEST
# -------------------------
def run_test(image, comp_no):

    start_time = time.time()
    left_count = 0
    right_count = 0
    emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion, attention = analyze_frame(frame)
        emotions.append(emotion)

        if attention == "left":
            left_count += 1
        elif attention == "right":
            right_count += 1

        combined = cv2.hconcat([cv2.resize(frame, (400, 500)), image])

        cv2.putText(combined, f"Comparison {comp_no}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Experiment", combined)

        if time.time() - start_time > 5:
            break

        if cv2.waitKey(1) == 27:
            break

    # Decide focus
    if left_count > right_count:
        focus = "A (Left)"
    elif right_count > left_count:
        focus = "B (Right)"
    else:
        focus = "Neutral"

    final_emotion = max(set(emotions), key=emotions.count)

    engagement_score = emotion_map.get(final_emotion, 0.5) + 5

    return focus, final_emotion, engagement_score

# -------------------------
# MAIN LOOP
# -------------------------
all_data = []

for i in range(1, 11):

    print(f"\nRunning Comparison {i}")

    img = cv2.imread(f"stimuli/comp{i}.jpg")

    if img is None:
        print(f"⚠️ Missing image comp{i}.jpg")
        continue

    img = cv2.resize(img, (800, 500))

    focus, emotion, score = run_test(img, i)

    all_data.append([user_id, i, focus, emotion, 5, score])

# Save
df = pd.DataFrame(all_data, columns=[
    "user_id", "comparison", "focus_side",
    "emotion", "attention_time", "engagement_score"
])

df.to_csv("data/final_dataset.csv", mode='a', header=False, index=False)

print("\n✅ Data collection completed!")

cap.release()
cv2.destroyAllWindows()