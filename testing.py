import requests
import numpy as np
import torch
import nibabel as nib
import torchio as tio
from loses import compute_losses

def validate_response(response):
    try:
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("Response is not a dictionary")
        if "prediction" not in result:
            raise KeyError("Response missing 'prediction' key")
        if not isinstance(result["prediction"], list):
            raise ValueError("'prediction' must be a list")
        expected_loss_keys = ["total_loss", "bce_loss", "dice_loss", "iou_loss"]
        for key in expected_loss_keys:
            if key in result and not isinstance(result[key], (int, float)):
                raise ValueError(f"'{key}' must be a number, got {type(result[key])}")
        return result
    except requests.exceptions.JSONDecodeError:
        raise ValueError("Invalid JSON response")
    except Exception as e:
        raise ValueError(f"Response validation failed: {str(e)}")

def load_true_mask(mask_path):
    try:
        transform = tio.Compose([
            tio.Resample(target=(1, 1, 1)),
        ])
        mask_img = nib.load(mask_path)
        mask_array = np.array(mask_img.get_fdata())
        if mask_array.ndim == 3:
            mask_array = mask_array[np.newaxis, ...]
        mask_subject = tio.Subject(image=tio.ScalarImage(tensor=mask_array))
        transformed = transform(mask_subject)
        mask_array = transformed.image.numpy()
        if mask_array.shape[0] > 1:
            mask_array = mask_array[0:1, ...]
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32)
        mask_tensor = (mask_tensor > 0.5).float()
        return mask_tensor
    except Exception as e:
        raise ValueError(f"Failed to load true mask: {str(e)}")

def save_response_mask(predicted_mask, output_path):
    try:
        predicted_mask = torch.tensor(predicted_mask, dtype=torch.float32)
        if predicted_mask.dim() == 5:
            predicted_mask = predicted_mask.squeeze(0)
        predicted_mask = (predicted_mask > 0.5).float()
        predicted_mask_np = predicted_mask.cpu().numpy()
        nii_mask = nib.Nifti1Image(predicted_mask_np[0], affine=np.eye(4))
        nib.save(nii_mask, output_path)
        return predicted_mask
    except Exception as e:
        raise ValueError(f"Failed to save response predicted mask: {str(e)}")

def compare_masks(server_mask_path, response_mask_path):
    try:
        server_mask = nib.load(server_mask_path).get_fdata()
        response_mask = nib.load(response_mask_path).get_fdata()
        if server_mask.shape != response_mask.shape:
            raise ValueError(f"Mask shape mismatch: server {server_mask.shape} vs response {response_mask.shape}")
        is_identical = np.array_equal(server_mask, response_mask)
        mean_abs_diff = np.mean(np.abs(server_mask - response_mask))
        return is_identical, mean_abs_diff
    except Exception as e:
        raise ValueError(f"Failed to compare masks: {str(e)}")

def recompute_losses(predicted_mask, true_mask):
    try:
        if predicted_mask.shape != true_mask.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_mask.shape} vs true {true_mask.shape}")
        total_loss, bce, dice, iou = compute_losses(predicted_mask, true_mask)
        return {
            "total_loss": total_loss.item(),
            "bce_loss": bce.item(),
            "dice_loss": dice.item(),
            "iou_loss": iou.item()
        }
    except Exception as e:
        raise ValueError(f"Failed to recompute losses: {str(e)}")

def compare_losses(server_losses, recomputed_losses):
    try:
        results = {}
        for key in ["total_loss", "bce_loss", "dice_loss", "iou_loss"]:
            if key in server_losses:
                server_val = server_losses[key]
                recomputed_val = recomputed_losses[key]
                results[key] = abs(server_val - recomputed_val) <= 1e-5
        return results
    except Exception as e:
        raise ValueError(f"Failed to compare losses: {str(e)}")

def main():
    url = "http://localhost:8000/predict"
    mask_path = "inference_modalities/mask_patient1.nii"
    server_mask_path = "inference_modalities/server_predicted_mask.nii"
    response_mask_path = "inference_modalities/response_predicted_mask.nii"
    files = {
        "file1": ("modality0_patient1.nii", open("inference_modalities/modality0_patient1.nii", "rb"), "application/octet-stream"),
        "file2": ("modality1_patient1.nii", open("inference_modalities/modality1_patient1.nii", "rb"), "application/octet-stream"),
        "file3": ("modality2_patient1.nii", open("inference_modalities/modality2_patient1.nii", "rb"), "application/octet-stream"),
        "file4": ("modality3_patient1.nii", open("inference_modalities/modality3_patient1.nii", "rb"), "application/octet-stream"),
    }

    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        result = validate_response(response)

        true_mask = load_true_mask(mask_path)
        predicted_mask = save_response_mask(result["prediction"], response_mask_path)
        is_identical, mean_abs_diff = compare_masks(server_mask_path, response_mask_path)

        # Test Case 1: Predicted mask vs itself (should be zero loss)
        self_losses = recompute_losses(predicted_mask, predicted_mask)

        # Test Case 2: Predicted mask vs zero mask (should be high loss)
        zero_mask = torch.zeros_like(predicted_mask)
        zero_losses = recompute_losses(predicted_mask, zero_mask)

        # Test Case 3: Predicted mask vs flipped predicted mask (should be non-zero loss)
        flipped_mask = torch.flip(predicted_mask, dims=[-1])  # Flip along depth
        flipped_losses = recompute_losses(predicted_mask, flipped_mask)

        # Test Case 4: Predicted mask vs true mask
        recomputed_losses = recompute_losses(predicted_mask, true_mask)
        server_losses = {k: result[k] for k in ["total_loss", "bce_loss", "dice_loss", "iou_loss"] if k in result}
        loss_matches = compare_losses(server_losses, recomputed_losses)

        # Print results
        print(f"Test Case 1 (Predicted vs Predicted): {self_losses}")
        print(f"Test Case 2 (Predicted vs Zero): {zero_losses}")
        print(f"Test Case 3 (Predicted vs Flipped): {flipped_losses}")
        print(f"Test Case 4 (Predicted vs True): {recomputed_losses}")
        print(f"Server vs Recomputed Matches: {loss_matches}")
        print(f"Masks Identical: {is_identical}, Mean Absolute Difference: {mean_abs_diff}")

    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {str(e)}")
    except ValueError as e:
        print(f"Validation or processing error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()