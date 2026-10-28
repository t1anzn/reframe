import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2 

# Test image name   
name = "room_01"

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
col, row = np.meshgrid(np.arange(w), np.arange(h)) 

# Backproject: each pixel becomes a 3D point using its depth and the camera intrinsics (f and the image centre)
point_x = (col - w / 2) * depth / f # left-right position in 3D space
point_y = (row - h / 2) * depth / f # up-down position in 3D space
point_z = depth # distance from the camera

# Identity test: project the 3D points straight back to pixels
col_back = f * point_x / point_z + w / 2
row_back = f * point_y / point_z + h / 2
print("max error:", np.abs(col_back - col).max(), np.abs(row_back - row).max()) # expect near 0

# Sideways shift: move the camera right by s (same units as depth)
camera_shift = 0.1
point_x_moved = point_x - camera_shift # every point is further left as seen from the moved camera
col_new = f * point_x_moved / point_z + w / 2  # where the points end up in the new image

pixels_moved = col - col_new # pixels each point moved to the left
print("near pixels moved:", pixels_moved[depth < 1.5].mean())
print("far pixels moved:", pixels_moved[depth > 8].mean()) 

# Round the new columns to whole pixels and keep only points that land inside the image
col_dst = np.round(col_new).astype(int)
inside = (col_dst >= 0) & (col_dst < w) # keep only points that land inside the image

# One entry per point that stays in view
colours = image[inside] # colour of each point
col_kept = col_dst[inside] # its new column
row_kept = row[inside] # its row (rows do not change in a sideways shift)
dist_kept = point_z[inside] # its distance from the camera

# Z-buffer: for every new pixel, remember the smallest distance that landed there
zbuf = np.full((h, w), np.inf) # Initialize the z-buffer with infinity (no points have landed yet)
np.minimum.at(zbuf, (row_kept, col_kept), dist_kept) # For each point, update the z-buffer at its new pixel with the minimum distance (nearest point)

# A point wins if it is the nearest one at its new pixel
win = dist_kept == zbuf[row_kept, col_kept]

shifted = np.zeros_like(image) # starts black
shifted[row_kept[win], col_kept[win]] = colours[win]

hole = np.isinf(zbuf) # pixels where nothing landed

# Plot the original image, the shifted image and the hole mask
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(shifted)
ax[1].set_title("Camera moved right")
ax[2].imshow(hole, cmap="gray")
ax[2].set_title("Hole mask (white = hole)")
plt.show()