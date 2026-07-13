import pandas as pd
import random

data = []

# Generate 1000 sample users
for i in range(1000):
    emotion = random.choice([
        "happy", "sad", "angry", "neutral", "surprise"
    ])
    
    data.append([i+1, emotion])

df = pd.DataFrame(data, columns=["user_id", "emotion"])
df.to_csv("datasets/facial_emotion.csv", index=False)

print("✅ Facial dataset generated successfully!")