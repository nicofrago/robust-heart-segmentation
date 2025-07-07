import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F



def vis2(im, pim):
    im = im[0][0].detach().cpu().numpy()
    pim = pim[0][0].detach().cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(im, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(pim, cmap='gray')
    ax[1].set_title("Perturbed Image")
    plt.tight_layout()
    plt.show()

def get_data_min_max(dataset):
    minv = np.inf
    maxv = -np.inf
    for i, (im, lb) in enumerate(dataset):
        minv = min(minv, im.min())
        maxv = max(maxv, im.max())
    return minv, maxv

def visualize_segmentations(network, data_loader, device, max_samples=5):
    network.eval()
    samples_displayed = 0

    with torch.no_grad():
        for scans, masks in data_loader:
            scans = scans.to(device)
            masks = masks.to(device)

            predictions = network(scans)
            seg_results = torch.argmax(predictions, dim=1)

            for idx in range(scans.size(0)):
                if samples_displayed >= max_samples:
                    return

                scan_slice = scans[idx].cpu().squeeze()
                true_mask = masks[idx].cpu()
                pred_mask = seg_results[idx].cpu()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(scan_slice, cmap='gray')
                axs[0].set_title('Scan Slice')
                axs[1].imshow(true_mask, cmap='tab20')
                axs[1].set_title('True Segmentation')
                axs[2].imshow(pred_mask, cmap='tab20')
                axs[2].set_title('Network Output')
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()

                samples_displayed += 1
