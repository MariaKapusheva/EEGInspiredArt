import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import numpy as np 
from sklearn.cluster import KMeans

PROMPTS_FILE = "EEGInspiredArt/eeg_prompts.json"
IMAGES_DIR = "EEGInspiredArt/generated_images"
RESULTS_FILE = "EEGInspiredArt/evaluation_scores.json"
CLIP_NAME = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_RGB = {
    "deep red": (180, 0, 0),
    "orange": (255, 140, 0),
    "yellow": (255, 230, 50),
    "green": (0, 180, 0),
    "blue": (50, 80, 255),
    "violet": (160, 0, 200)
}

print(f"Loading CLIP model '{CLIP_NAME}' on {DEVICE}...") # loading the clip model from kaggle
clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)

with open(PROMPTS_FILE, "r") as f:
    prompts = json.load(f)
print(f"Loaded {len(prompts)} prompts.")

def get_dominant_colors(image, k=3):
    """Extract dominant colors via k-means clustering."""
    img = image.resize((128, 128))
    arr = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=3, random_state=42)
    kmeans.fit(arr)
    return [tuple(map(int, c)) for c in kmeans.cluster_centers_]

def color_distance(c1, c2):
    """Euclidean distance between two RGB colors."""
    return np.linalg.norm(np.array(c1) - np.array(c2))

def color_match_score(target_color_label, image):
    """Compare target EEG color label to actual image colors."""
    if target_color_label not in COLOR_RGB:
        return None
    target_rgb = np.array(COLOR_RGB[target_color_label])
    dominant_colors = get_dominant_colors(image, k=5)
    # Find smallest distance between image color and target
    distances = [color_distance(target_rgb, dc) for dc in dominant_colors]
    min_dist = min(distances)
    # Normalizing because we want the similarity score in the [0,1] range, not just the distance 
    score = 1 - (min_dist / 441)
    return max(0, min(1, score))

# Computing CLIP similarity
results = {}
for fname, data in tqdm(prompts.items(), desc="Computing CLIP similarity"):
    prompt = data["prompt"]
    target_color = data.get("color", None)
    img_path = os.path.join(IMAGES_DIR, fname.replace(".npy", ".png"))

    if not os.path.exists(img_path):
        print(f"Skipping {fname}, image not found.")
        continue

    image = Image.open(img_path).convert("RGB")

    # Preparing inputs for CLIP
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        # caclulating the cosine similarity between text and image embeddings
        clip_score = F.cosine_similarity(text_embeds, image_embeds).item()

    color_score = color_match_score(target_color, image)

    results[fname] = {
        "prompt": prompt,
        "intended_color": target_color,
        "clip_score": clip_score,
        "color_match_score": color_score
    }


with open(RESULTS_FILE, "w") as f: # saving results
    json.dump(results, f, indent=4)

# average results
clip_vals = [r["clip_score"] for r in results.values()]
avg_clip = np.mean(clip_vals)

color_vals = [r["color_match_score"] for r in results.values() if r["color_match_score"] is not None]
avg_color = np.mean(color_vals) if color_vals else None

print(f"Average CLIP similarity: {avg_clip:.4f}")
print(f"Average color match score: {avg_color:.4f}")

