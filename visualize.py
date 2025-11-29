import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import torch

def generate_slice_overlays(images, predicted_mask):
    try:
        # Ensure images and mask are numpy arrays
        images = images.cpu().numpy() if isinstance(images, torch.Tensor) else images
        predicted_mask = predicted_mask.cpu().numpy() if isinstance(predicted_mask, torch.Tensor) else predicted_mask

        # Handle batch dimension in images: [1, C, H, W, D] -> [C, H, W, D]
        if images.ndim == 5:
            images = images[0]  # Remove batch dimension

        # Ensure correct shapes: images [C, H, W, D], predicted_mask [1, H, W, D]
        if predicted_mask.ndim == 5:
            predicted_mask = predicted_mask[0]  # [1, H, W, D]
        if predicted_mask.ndim == 3:
            predicted_mask = predicted_mask[np.newaxis, ...]  # [1, H, W, D]

        # Middle slices for axial, sagittal, coronal views
        h, w, d = images.shape[1:]  # [C, H, W, D]
        axial_idx = d // 2
        sagittal_idx = w // 2
        coronal_idx = h // 2

        # Initialize result dictionary for base64-encoded images
        overlays = {}

        # Process each modality (channel) in images
        for channel in range(images.shape[0]):
            # Axial view (slice along depth)
            fig, ax = plt.subplots(figsize=(5, 5))
            scan_slice = images[channel, :, :, axial_idx]
            mask_slice = predicted_mask[0, :, :, axial_idx]
            ax.imshow(scan_slice, cmap="gray")
            ax.imshow(mask_slice, cmap="jet", alpha=0.5)
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            overlays[f"axial_modality{channel}"] = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Sagittal view (slice along width)
            fig, ax = plt.subplots(figsize=(5, 5))
            scan_slice = images[channel, :, sagittal_idx, :]
            mask_slice = predicted_mask[0, :, sagittal_idx, :]
            ax.imshow(scan_slice, cmap="gray")
            ax.imshow(mask_slice, cmap="jet", alpha=0.5)
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            overlays[f"sagittal_modality{channel}"] = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Coronal view (slice along height)
            fig, ax = plt.subplots(figsize=(5, 5))
            scan_slice = images[channel, coronal_idx, :, :]
            mask_slice = predicted_mask[0, coronal_idx, :, :]
            ax.imshow(scan_slice, cmap="gray")
            ax.imshow(mask_slice, cmap="jet", alpha=0.5)
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            overlays[f"coronal_modality{channel}"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        return overlays
    except Exception as e:
        raise ValueError(f"Failed to generate slice overlays: {str(e)}")