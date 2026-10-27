import numpy as np
from PIL import Image, ImageOps
import cv2 

# Test image name   
name = "person_03"

# Load the image and preprocess it
image = Image.open(f"./data/photos/{name}.png")
image = ImageOps.exif_transpose(image).convert("RGB")
image.thumbnail((1024, 1024)) # Resize the image
image = np.array(image) # Convert to a NumPy array

# Load the depth map from the .npy file
raw_depth = np.load(f"./outputs/{name}_depth.npy")
#print("depth shape:", raw_depth.shape)

# Make the depth map the same size as the image
h, w = image.shape[:2] # [576, 1024, 3] -> [576, 1024]
print("image:", (h, w), "depth:", raw_depth.shape)
if raw_depth.shape != (h, w): 
    raw_depth = cv2.resize(raw_depth, (w, h))

# Subtract the minimum value so the depth map starts at 0
# Scale the depth map to the range [0, 1] for visualization
d = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min())

# Invert the depth map to get a distance map
# d is 1 for the nearest pixel and 0 for the farthest. Flip it into a distance: near becomes small and far becomes large
depth = 1.0 / (d + 0.1) # Invert the depth map and add a small value to avoid division by zero
print("depth range:", depth.min(), depth.max()) # 1-10

#------
# Convert points to 3D points
fov_deg = 60 # Guessing the field of view in degrees 

# Original: fov = 2 * arctan(w / (2 * f))
# Solved for f: f = (w / 2) / tan(fov / 2)
# Focal length in pixels
f = (w / 2) / np.tan(np.radians(fov_deg) / 2)

# Create a grid of pixel coordinates
u, v = np.meshgrid(np.arange(w), np.arange(h)) 

# Backproject: each pixel becomes a 3D point using its depth and the camera intrinsics (f and the image centre)
X = (u - w / 2) * depth / f # left-right position in 3D space
Y = (v - h / 2) * depth / f # up-down position in 3D space
Z = depth # distance from the camera

# Identity test: project the 3D points straight back to pixels
u2 = f * X / Z + w / 2
v2 = f * Y / Z + h / 2
print("max error:", np.abs(u2 - u).max(), np.abs(v2 - v).max()) # expect near 0

