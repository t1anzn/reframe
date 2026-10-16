from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import pipeline

# Pick the best available device: NVIDIA GPU, Apple GPU (MPS) or CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("using", device)

# Load the depth estimation model using the Hugging Face pipeline
depth_model = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=device,
)
# Create the output directory if it doesn't exist
out_dir = Path("outputs") 
out_dir.mkdir(exist_ok=True) 

# Possible image file extensions to process
extensions = {".jpg", ".jpeg", ".png"}

for photo in sorted(Path("data/photos").iterdir()):
    if photo.suffix.lower() not in extensions:
        continue # Skips invalid file extensions

    image = Image.open(photo) # Open the image file

    # Depth Anything 
    image = ImageOps.exif_transpose(image).convert("RGB") # Correct orientation and convert to RGB (3 channels) for consistency
    image.thumbnail((1024, 1024)) # Resize the image to a maximum of 1024x1024 pixels while maintaining aspect ratio
    result = depth_model(image)

    result["depth"].save(out_dir / f"{photo.stem}_depth.png") # Save the depth map as a PNG file in the output directory as "{original_filename}_depth.png"
    np.save(out_dir / f"{photo.stem}_depth.npy", result["predicted_depth"].squeeze().cpu().numpy()) # Save the predicted depth as a NumPy array in the output directory as "{original_filename}_depth.npy"
    print(f"Depth map saved for {photo.name} as {photo.stem}_depth.png") 

    
