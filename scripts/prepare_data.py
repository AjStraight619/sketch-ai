import ndjson
import numpy as np
import os
from PIL import Image, ImageDraw
import random
from tqdm import tqdm

SOURCE_DIR = "data/raw"
TARGET_DIR = "data/images"
SAMPLES_PER_CLASS = 3000
IMAGE_SIZE = 28

os.makedirs(TARGET_DIR, exist_ok=True)

def render_strokes(strokes, size=28, lw=3):
    img = Image.new("L", (size, size), "white")
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        xs, ys = stroke
        points = list(zip(xs, ys))
        if len(points) > 1:
            draw.line(points, fill=0, width=lw)
    return np.array(img)

for fname in tqdm(os.listdir(SOURCE_DIR)):
    if not fname.endswith(".ndjson"):
        continue
    class_name = fname.replace(".ndjson", "")
    with open(os.path.join(SOURCE_DIR, fname), "r") as f:
        drawings = ndjson.load(f)
    subset = random.sample(drawings, min(SAMPLES_PER_CLASS, len(drawings)))
    images = []
    for sample in subset:
        strokes = sample["drawing"]
        # Normalize strokes to IMAGE_SIZE x IMAGE_SIZE
        all_x = [x for stroke in strokes for x in stroke[0]]
        all_y = [y for stroke in strokes for y in stroke[1]]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        norm_strokes = [
            [
                [int((x - min_x) / max(1, max_x - min_x) * (IMAGE_SIZE - 1)) for x in stroke[0]],
                [int((y - min_y) / max(1, max_y - min_y) * (IMAGE_SIZE - 1)) for y in stroke[1]]
            ]
            for stroke in strokes
        ]
        img_arr = render_strokes(norm_strokes, size=IMAGE_SIZE)
        images.append(img_arr)
    # Save as numpy array
    images_np = np.stack(images)
    np.save(os.path.join(TARGET_DIR, f"{class_name}.npy"), images_np)
    print(f"Saved {images_np.shape[0]} images for class '{class_name}'.")

print("Done converting all classes!")
