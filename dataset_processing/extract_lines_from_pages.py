import cv2
import os
from segment import segment_lines

INPUT_DIR = "raw_pages"
OUTPUT_DIR = "../style_learning/style_dataset/writer_me"

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for fname in os.listdir(INPUT_DIR):
    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img is None:
        continue

    lines = segment_lines(img)

    for line in lines:
        h, w, _ = line.shape
        if h < 20 or w < 100:   # filter noise
            continue

        out = os.path.join(OUTPUT_DIR, f"line_{count:04d}.png")
        cv2.imwrite(out, line)
        count += 1

print(f"Saved {count} line images")
