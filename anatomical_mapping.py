import numpy as np
from nilearn import datasets
import nibabel as nib
from scipy.ndimage import label, center_of_mass
import kagglehub

# Download latest version
path = kagglehub.dataset_download("wenny5/anatomic-atlas")

def analyze_and_map_mask(mask, voxel_volume_mm3=1.0):
    try:
        # Fetch atlases
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')
        atlas_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-1mm')
        atlas_nifti = atlas['maps']
        atlas_data = atlas_nifti.get_fdata()
        atlas_affine = atlas_nifti.header.get_qform()
        inv_affine = np.linalg.inv(atlas_affine)
        atlas_sub_data = atlas_sub['maps'].get_fdata()

        jhu_nifti = nib.load(path + "/JHU-ICBM-tracts-maxprob-thr0-1mm.nii")
        jhu_data = jhu_nifti.get_fdata()
        jhu_inv_affine = np.linalg.inv(jhu_nifti.header.get_qform())

        # JHU ICBM-DTI-81 labels
        jhu_region_map = {
            0: "Background",
            1: "Middle cerebellar peduncle",
            2: "Pontine crossing tract",
            3: "Genu of corpus callosum",
            4: "Body of corpus callosum",
            5: "Splenium of corpus callosum",
            6: "Fornix",
            7: "Corticospinal tract R",
            8: "Corticospinal tract L",
            9: "Medial lemniscus R",
            10: "Medial lemniscus L",
            11: "Inferior cerebellar peduncle R",
            12: "Inferior cerebellar peduncle L",
            13: "Superior cerebellar peduncle R",
            14: "Superior cerebellar peduncle L",
            15: "Cerebral peduncle R",
            16: "Cerebral peduncle L",
            17: "Anterior limb of internal capsule R",
            18: "Anterior limb of internal capsule L",
            19: "Posterior limb of internal capsule R",
            20: "Posterior limb of internal capsule L",
            21: "Retrolenticular part of internal capsule R",
            22: "Retrolenticular part of internal capsule L",
            23: "Anterior corona radiata R",
            24: "Anterior corona radiata L",
            25: "Superior corona radiata R",
            26: "Superior corona radiata L",
            27: "Posterior corona radiata R",
            28: "Posterior corona radiata L",
            29: "Posterior thalamic radiation R",
            30: "Posterior thalamic radiation L",
            31: "Sagittal stratum R",
            32: "Sagittal stratum L",
            33: "External capsule R",
            34: "External capsule L",
            35: "Cingulum (cingulate gyrus) R",
            36: "Cingulum (cingulate gyrus) L",
            37: "Cingulum (hippocampus) R",
            38: "Cingulum (hippocampus) L",
            39: "Fornix (cres) R",
            40: "Fornix (cres) L",
            41: "Superior longitudinal fasciculus R",
            42: "Superior longitudinal fasciculus L",
            43: "Superior fronto-occipital fasciculus R",
            44: "Superior fronto-occipital fasciculus L",
            45: "Uncinate fasciculus R",
            46: "Uncinate fasciculus L",
            47: "Tapetum R",
            48: "Tapetum L",
            101: "Periventricular White Matter",
            102: "Subcortical White Matter",
            103: "Juxtacortical"
        }

        # Harvard-Oxford region mapping
        region_map = {0: "Background"}
        for idx, label_name in enumerate(atlas['labels']):
            region_map[idx + 1] = label_name
        for idx, label_name in enumerate(atlas_sub['labels']):
            region_map[idx + 11] = label_name
        region_map.update({
            101: "Periventricular White Matter",
            102: "Subcortical White Matter",
            103: "Juxtacortical"
        })

        # Transform centroids to MNI space
        mask_affine = np.array([[-1, 0, 0, 90], [0, -1, 0, 126], [0, 0, 1, -72], [0, 0, 0, 1]])

        def transform_centroid(centroid):
            return np.dot(mask_affine, np.array(list(centroid) + [1]))[:3]

        def map_centroid_to_region(centroid):
            centroid_mni = transform_centroid(centroid)
            # Check MNI bounds
            if not (-90 <= centroid_mni[0] <= 90 and -126 <= centroid_mni[1] <= 90 and -72 <= centroid_mni[2] <= 108):
                return "Out of brain bounds"
            # Try JHU atlas
            centroid_voxel = np.dot(jhu_inv_affine, np.array(list(centroid_mni) + [1]))[:3]
            centroid_voxel = np.round(centroid_voxel).astype(int)
            centroid_voxel = np.clip(centroid_voxel, [0, 0, 0], np.array(jhu_data.shape) - 1)
            region_value = jhu_data[tuple(centroid_voxel)]
            if region_value > 0:
                return jhu_region_map.get(int(region_value), f"Unknown JHU value {int(region_value)}")
            # Fallback to Harvard-Oxford
            centroid_voxel = np.dot(inv_affine, np.array(list(centroid_mni) + [1]))[:3]
            centroid_voxel = np.round(centroid_voxel).astype(int)
            centroid_voxel = np.clip(centroid_voxel, [0, 0, 0], np.array(atlas_data.shape) - 1)
            region_value = atlas_data[tuple(centroid_voxel)]
            if region_value > 0:
                return region_map.get(int(region_value), f"Unknown Cortical value {int(region_value)}")
            region_value = atlas_sub_data[tuple(centroid_voxel)]
            # Periventricular check
            ventricle_center = np.array([0, 0, 0])
            distance = np.linalg.norm(centroid_mni - ventricle_center)
            if distance < 15 and region_value == 0:
                return "Periventricular White Matter"
            # Subcortical/juxtacortical fallback
            if region_value == 0:
                return "Subcortical White Matter" if distance < 30 else "Juxtacortical"
            return region_map.get(int(region_value) + 10, f"Unknown Subcortical value {int(region_value)}")

        # Analyze mask
        structure = np.ones((3, 3, 3))
        labeled, num_features = label(mask, structure=structure)
        regions = {}
        for region_idx in range(1, num_features + 1):
            region_voxels = (labeled == region_idx)
            region_size = region_voxels.sum()
            if region_size > 1:
                centroid = center_of_mass(mask, labeled, region_idx)
                volume_cm3 = region_size * voxel_volume_mm3 / 1000  # Convert to cm³
                region_name = map_centroid_to_region(centroid)
                regions[region_idx] = {
                    "volume_cm3": round(volume_cm3, 3),
                    "location": region_name
                }

        return regions
    except Exception as e:
        raise ValueError(f"Failed to analyze and map mask: {str(e)}")