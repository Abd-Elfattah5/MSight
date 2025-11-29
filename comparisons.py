import numpy as np
import nibabel as nib
import logging
from loses import compute_losses
import torch
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def compare_nii_files(file1_path, file2_path):
    try:
        nii1 = nib.load(file1_path).get_fdata()
        nii2 = nib.load(file2_path).get_fdata()
        logger.debug(f"File 1 shape: {nii1.shape}")
        logger.debug(f"File 2 shape: {nii2.shape}")
        if nii1.shape != nii2.shape:
            raise ValueError(f"Shape mismatch: {nii1.shape} vs {nii2.shape}")
        is_identical = np.array_equal(nii1, nii2)
        mean_abs_diff = np.mean(np.abs(nii1 - nii2))
        total_loses = compute_losses(torch.tensor(nii1), torch.tensor(nii2))
        logger.info(f"Masks identical: {is_identical}")
        logger.info(f"total loses: {total_loses}")
        logger.info(f"Mean absolute difference: {mean_abs_diff}")
        logger.debug(f"Sample slice file 1: {nii1[:, :, min(nii1.shape[2]//2, nii1.shape[2]-1)][:5, :5]}")
        logger.debug(f"Sample slice file 2: {nii2[:, :, min(nii2.shape[2]//2, nii2.shape[2]-1)][:5, :5]}")
        return is_identical, mean_abs_diff
    except Exception as e:
        logger.error(f"Error comparing files: {str(e)}")
        raise

if __name__ == "__main__":
    file1_path = "/teamspace/studios/this_studio/inference_modalities/mask_patient1.nii"
    file2_path = "/teamspace/studios/this_studio/inference_modalities/response_predicted_mask.nii"
    compare_nii_files(file1_path, file2_path)