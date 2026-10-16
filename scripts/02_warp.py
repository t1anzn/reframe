import numpy as np
from PIL import Image, ImageOps

# Test image name   
name = "person_03"

# Load the image and preprocess it
image = Image.open(f"./data/photos/{name}.png")
image = ImageOps.exif_transpose(image).convert("RGB")
image.thumbnail((1024, 1024)) # Resize the image
image = np.array(image) # Convert to a NumPy array

# Load the depth map from the .npy file
disp = np.load(f"./outputs/{name}_depth.npy")
#print("depth shape:", disp.shape)

