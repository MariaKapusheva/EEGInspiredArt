import os
import json
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torch

PROMPTS_FILE = "eeg_prompts.json"
OUTPUT_DIR = "generated_images"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"   
IMAGE_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading model '{MODEL_NAME}' on {DEVICE}...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))  # optional: disable filter for abstract art

with open(PROMPTS_FILE, "r") as f:
    prompts = json.load(f)

print(f"Loaded {len(prompts)} prompts.")

for fname, data in tqdm(prompts.items(), desc="Generating images"):
    prompt = data["prompt"]
    out_path = os.path.join(OUTPUT_DIR, fname.replace(".npy", ".png"))

    if os.path.exists(out_path):
        continue  

    image = pipe(prompt, guidance_scale=7.5).images[0]
    image.save(out_path)

print(f"All images saved to '{OUTPUT_DIR}/'")
