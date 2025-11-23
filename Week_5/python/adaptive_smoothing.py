# adaptive_smoothing.py
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

def bilateral_example(input_path, out_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # bilateral filter: d=9, sigmaColor=75, sigmaSpace=75
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite(out_path, smooth)
    # save side-by-side
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title('Noisy'); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(smooth, cmap='gray'); plt.title('Bilateral'); plt.axis('off')
    plt.tight_layout()
    plt.savefig("figures/fig_smoothing.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    # Put a noisy image in data/noisy.png or use any sample
    bilateral_example("data/noisy.png", "figures/smoothed.png")
