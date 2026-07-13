import pandas as pd

eye_df = pd.read_csv("datasets/eye_tracking.csv")
face_df = pd.read_csv("datasets/facial_emotion.csv")

print("Eye shape:", eye_df.shape)
print("Face shape:", face_df.shape)

if len(eye_df) == 0 or len(face_df) == 0:
    print("❌ One dataset is empty!")
    exit()

# Match size
face_df = face_df.sample(n=len(eye_df), replace=True).reset_index(drop=True)
eye_df = eye_df.reset_index(drop=True)

merged_df = pd.concat([face_df, eye_df], axis=1)

# Emotion score
emotion_map = {
    "happy": 1,
    "surprise": 0.8,
    "neutral": 0.5,
    "sad": 0.2,
    "angry": 0
}

merged_df["emotion_score"] = merged_df["emotion"].map(emotion_map)

merged_df["engagement_score"] = (
    merged_df["emotion_score"] + merged_df["attention_time"]
)

merged_df.to_csv("data/merged_dataset.csv", index=False)

print("✅ Merge successful!")