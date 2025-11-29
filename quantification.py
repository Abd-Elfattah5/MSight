from scipy.ndimage import label, center_of_mass
import plotly.graph_objs as go
import numpy as np

def analyze_mask(mask, connectivity=3):
    structure = np.ones((3,3,3)) if connectivity == 3 else None
    labeled, num_features = label(mask, structure=structure)

    regions = {}
    for region_idx in range(1, num_features+1):
        region_voxels = (labeled == region_idx)
        region_size = region_voxels.sum()
        print(f"Raw Region {region_idx}: Size = {region_size}")  # Debug print
        if region_size > 1:
            centroid = center_of_mass(mask, labeled, region_idx)
            regions[region_idx] = {
                "volume": int(region_size),
                "centroid": tuple(np.round(centroid, 2))
            }

    return labeled, regions