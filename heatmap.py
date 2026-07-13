import pandas as pd
import numpy as np
import cv2

df = pd.read_csv("data/merged_dataset.csv")

heatmap = np.zeros((500, 800), dtype=np.float32)

for _, row in df.iterrows():
    x = int(row["gaze_x"])
    y = int(row["gaze_y"])

    if 0 <= x < 800 and 0 <= y < 500:
        heatmap[y, x] += 1

heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

heatmap = heatmap.astype(np.uint8)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

cv2.imwrite("data/heatmap.jpg", heatmap_color)

print("🔥 Heatmap generated")