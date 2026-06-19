import matplotlib
matplotlib.use("Qt5Agg")   # BEST interactive backend on macOS


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift
from skimage.io import imread, imsave
import os

import random

N_images = 10000
output_dir = "GRMHD_augmented"
os.makedirs(output_dir, exist_ok=True)

# Load base image
img = imread("m87.png").astype(float)
if img.ndim == 3:
    img = img.mean(axis=2)

img /= np.max(img)

ny, nx = img.shape
Y, X = np.indices((ny, nx))
cx, cy = nx/2, ny/2
r = np.sqrt((X-cx)**2 + (Y-cy)**2)
theta = np.arctan2(Y-cy, X-cx)

r_norm = r / np.max(r)

for i in range(N_images):

    # -------- Physical Parameters --------
    spin = np.random.uniform(0.0, 0.99)
    inclination = np.random.uniform(10, 40) * np.pi/180
    boost_amp = 0.3 + 0.5*np.sin(inclination)
    phi0 = np.random.uniform(-np.pi, np.pi)
    
    # -------- Doppler Boosting --------
    doppler = 1 + boost_amp * np.cos(theta - phi0)

    # -------- Optical Depth Variation --------
    tau = np.exp(- (r_norm - 0.5)**2 / (0.05 + 0.1*spin))
    opacity = np.exp(-tau)

    # -------- Spin-dependent Ring Shift --------
    radial_shift = 1 + 0.05*(spin - 0.5)
    warped_r = r_norm * radial_shift
    ring_mod = np.exp(-((warped_r - 0.5)**2)/(0.02 + 0.05*(1-spin)))

    # -------- Turbulent Magnetic Fluctuations --------
    turbulence = gaussian_filter(np.random.normal(0, 0.05, img.shape), 3)

    # -------- Combine --------
    new_img = img * doppler * opacity * ring_mod + turbulence

    # Smooth slightly (finite beam)
    new_img = gaussian_filter(new_img, sigma=1.0)

    # Normalize
    new_img = np.clip(new_img, 0, None)
    new_img /= np.max(new_img)

    imsave(f"{output_dir}/m87_grmhd_{i:04d}.png",
           (new_img*255).astype(np.uint8))

print("GRMHD-inspired dataset generated.")



# Get all generated files
files = [f for f in os.listdir(output_dir) if f.endswith(".png")]

# Randomly select two
random_files = random.sample(files, 2)

# Plot
plt.figure(figsize=(8,4))

for i, fname in enumerate(random_files):
    img = imread(os.path.join(output_dir, fname))
    
    plt.subplot(1, 2, i+1)
    plt.imshow(img, cmap='afmhot')
    plt.title(fname)
    plt.axis('off')

plt.tight_layout()
plt.show()
