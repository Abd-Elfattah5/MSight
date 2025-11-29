import nibabel as nib
import numpy as np
from scipy.ndimage import label, center_of_mass
import os
import pandas as pd
from datetime import datetime
from anatomical_mapping import analyze_and_map_mask
from report_generator import generate_report

def debug_lesion_analysis(mask_path, output_dir="/teamspace/studios/this_studio/inference_modalities/debug", voxel_volume_mm3=1.0):
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load predicted mask
        nii_mask = nib.load(mask_path)
        mask = nii_mask.get_fdata()
        if mask.ndim != 3:
            raise ValueError(f"Expected 3D mask, got shape {mask.shape}")

        # Analyze mask for Cartesian coordinates and sizes
        structure = np.ones((3, 3, 3))
        labeled, num_features = label(mask, structure=structure)
        lesion_data = []
        for region_idx in range(1, num_features + 1):
            region_voxels = (labeled == region_idx)
            region_size = region_voxels.sum()
            if region_size > 1:  # Ignore single-voxel regions
                centroid = center_of_mass(mask, labeled, region_idx)
                volume_cm3 = region_size * voxel_volume_mm3 / 1000
                lesion_data.append({
                    "LesionID": region_idx,
                    "CentroidX": round(centroid[0], 3),
                    "CentroidY": round(centroid[1], 3),
                    "CentroidZ": round(centroid[2], 3),
                    "SizeVoxels": region_size,
                    "VolumeCm3": round(volume_cm3, 3)
                })

        # Map to anatomical regions
        regions = analyze_and_map_mask(mask, voxel_volume_mm3=voxel_volume_mm3)
        for lesion in lesion_data:
            lesion_id = lesion["LesionID"]
            if lesion_id in regions:
                lesion["Location"] = regions[lesion_id]["location"]
                lesion["VolumeCm3"] = regions[lesion_id]["volume_cm3"]
            else:
                lesion["Location"] = "Not mapped"

        # Generate report
        patient_id = f"patient_{int(datetime.now().timestamp())}"
        report = generate_report(regions, patient_id=patient_id)

        # Save to text file
        text_output = f"Debugging Output - {timestamp}\n"
        text_output += f"Mask File: {mask_path}\n"
        text_output += f"Patient ID: {patient_id}\n\n"
        text_output += "Lesion Data:\n"
        for lesion in lesion_data:
            text_output += (f"Lesion {lesion['LesionID']}:\n"
                           f"  Centroid: ({lesion['CentroidX']}, {lesion['CentroidY']}, {lesion['CentroidZ']})\n"
                           f"  Size: {lesion['SizeVoxels']} voxels\n"
                           f"  Volume: {lesion['VolumeCm3']} cm³\n"
                           f"  Location: {lesion['Location']}\n\n")
        text_output += "Generated Report:\n"
        text_output += report

        text_file = os.path.join(output_dir, f"debug_lesion_analysis_{timestamp}.txt")
        with open(text_file, "w") as f:
            f.write(text_output)

        # Save to CSV
        df = pd.DataFrame(lesion_data)
        csv_file = os.path.join(output_dir, f"debug_lesion_analysis_{timestamp}.csv")
        df.to_csv(csv_file, index=False)

        print(f"Debugging output saved to:")
        print(f"  Text: {text_file}")
        print(f"  CSV: {csv_file}")

        return lesion_data, regions, report

    except Exception as e:
        print(f"Error in debug_lesion_analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage: replace with the actual mask path from server.py output
    mask_path = "/teamspace/studios/this_studio/inference_modalities/server_predicted_mask_1753307744.nii"
    debug_lesion_analysis(mask_path)