import litserve as ls
import torch
import nibabel as nib
import numpy as np
from model import UNet3D
import torchio as tio
from torch.utils.data import Dataset, DataLoader
from visualize import generate_slice_overlays
from anatomical_mapping import analyze_and_map_mask
from report_generator import generate_report
import asyncio
import time

class NiiDataset(Dataset):
    def __init__(self, nii_files):
        self.nii_files = nii_files
        self.transform = tio.Compose([
            tio.Resample(target=(1, 1, 1)),
        ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        try:
            channels = []
            for data in self.nii_files:
                nii_img = nib.Nifti1Image.from_bytes(data)
                img_array = np.array(nii_img.get_fdata())
                if img_array.ndim == 3:
                    img_array = img_array[np.newaxis, ...]
                subject = tio.Subject(image=tio.ScalarImage(tensor=img_array))
                transformed = self.transform(subject)
                img_array = transformed.image.numpy()
                channels.append(img_array)
            stacked = np.stack(channels, axis=0)
            stacked = stacked.squeeze(1)
            input_tensor = torch.tensor(stacked, dtype=torch.float32)
            return input_tensor
        except Exception as e:
            raise

class NeuroImagingAPI(ls.LitAPI):
    def __init__(self):
        super().__init__(max_batch_size=1)

    def setup(self, device):
        self.weights_path = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/farahmo/isbi-loocv-data/versions/17/unet_group1_epoch_112.pth"
        self.device = device
        self.model = UNet3D(in_channels=4, out_channels=1).to(device)
        try:
            self.model.load_state_dict(torch.load(self.weights_path, map_location=device))
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
        self.model.eval()

    async def decode_request(self, request):
        try:
            files = [await request.get(f"file{i}").read() for i in range(1, 5)]
            if any(file is None for file in files):
                raise ValueError("Four .nii files must be provided (file1, file2, file3, file4)")
            return files
        except Exception as e:
            raise

    def predict(self, data):
        try:
            files = asyncio.run(data)
            dataset = NiiDataset(files)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            for input_tensor in dataloader:
                input_tensor = input_tensor.to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    predicted_mask = (torch.sigmoid(outputs) > 0.5).float()
                    predicted_mask_np = predicted_mask.cpu().numpy()
                    nii_mask = nib.Nifti1Image(predicted_mask_np[0, 0], affine=np.eye(4))
                    timestamp = int(time.time())
                    nib.save(nii_mask, f"/teamspace/studios/this_studio/inference_modalities/server_predicted_mask_{timestamp}.nii")
                    overlays = generate_slice_overlays(input_tensor, predicted_mask)
                    regions = analyze_and_map_mask(predicted_mask_np[0, 0])
                    report = generate_report(regions, patient_id=f"patient_{timestamp}")
                response = {
                    "overlays": overlays,
                    "report": report
                }
                return response
        except Exception as e:
            raise

    def encode_response(self, output):
        try:
            return output
        except Exception as e:
            raise

if __name__ == "__main__":
    server = ls.LitServer(NeuroImagingAPI(), accelerator="cuda", timeout = 30*60)
    server.run(port=8000)