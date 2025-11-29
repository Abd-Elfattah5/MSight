import torch
from torch.utils.data import Dataset, DataLoader
import kagglehub

# Download latest version
path = kagglehub.dataset_download("farahmo/isbi-loocv-data")

print("Path to dataset files:", path)

class MSLesionDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = torch.load(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scans, mask, patient_id = self.data[idx]

        if self.transform:
            scans, mask = self.transform(scans, mask)

        return scans, mask