import matplotlib.pyplot as plt
import numpy as np
import torch

def denormalize_image(img):
    img = img * 0.5 + 0.5
    img = torch.clamp(img, 0, 1)
    return img

def show_prediction_overlay(results):
    for r in results:
        image = r["image"].squeeze()
        image = denormalize_image(image).numpy()
        mask = r["mask"].squeeze().numpy()
        pred = r["pred"].squeeze().numpy()
        plt.figure(figsize=(6,6))
        plt.imshow(image, cmap="gray")
        gt = np.zeros((*mask.shape,4))
        gt[...,1] = 1 # green for the gt
        gt[...,3] = mask * 0.4
        plt.imshow(gt)
        pr = np.zeros((*pred.shape,4))
        pr[...,0] = 1 # red for the prediction
        pr[...,3] = pred * 0.4
        plt.imshow(pr)
        plt.title(f"dice = {r['dice']:.3f}")
        plt.axis("off")
        plt.show()