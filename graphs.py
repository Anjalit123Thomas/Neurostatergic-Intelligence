import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/merged_dataset.csv")

if df.empty:
    print("❌ Dataset is empty! Cannot plot graphs.")
    exit()

if "emotion" not in df.columns:
    print("❌ 'emotion' column missing!")
    exit()

# Graph
df["emotion"].value_counts().plot(kind="bar")
plt.title("Emotion Distribution")
plt.savefig("data/emotion.png")

print("✅ Graph created!")