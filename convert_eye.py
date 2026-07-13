import pandas as pd
import random

data = []

# Create 1000 samples (same as face dataset)
for i in range(1000):
    user_id = i + 1
    
    gaze_x = random.randint(0, 800)
    gaze_y = random.randint(0, 500)
    
    attention_time = random.uniform(1, 5)  # more realistic

    data.append([user_id, gaze_x, gaze_y, attention_time])

df = pd.DataFrame(data, columns=[
    "user_id", "gaze_x", "gaze_y", "attention_time"
])

df.to_csv("datasets/eye_tracking.csv", index=False)

print("✅ Eye dataset created:", len(df))