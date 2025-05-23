import requests
import os
from tqdm import tqdm

CLASSES_TXT_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
NDJSON_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/raw"
NDJSON_OUT_DIR = "data/raw"

os.makedirs(NDJSON_OUT_DIR, exist_ok=True)

# Get all available Quick, Draw! classes
r = requests.get(CLASSES_TXT_URL)
all_words = [line.strip() for line in r.text.splitlines() if line.strip()]
print(f"Total words in Quick, Draw!: {len(all_words)}")

# Save word list
with open("data/words.txt", "w") as f:
    for w in all_words:
        f.write(w + "\n")
print("Saved full word list to data/words.txt")

# Download all .ndjson files 
for word in tqdm(all_words, desc="Downloading NDJSON files"):
    url = f"{NDJSON_BASE_URL}/{word}.ndjson"
    out_path = os.path.join(NDJSON_OUT_DIR, f"{word}.ndjson")
    if os.path.exists(out_path):
        continue
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(resp.content)
    else:
        print(f"FAILED to download: {url}")
print("All downloads complete!")



