import requests
import base64
import os

def save_overlays(overlays, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        for key, b64_string in overlays.items():
            with open(f"{output_dir}/{key}.png", "wb") as f:
                f.write(base64.b64decode(b64_string))
    except Exception as e:
        raise ValueError(f"Failed to save overlays: {str(e)}")

def main():
    url = "http://localhost:8000/predict"
    output_dir = "overlays"
    files = {
        "file1": ("modality0_patient0.nii", open("inference_modalities/modality0_patient0.nii", "rb"), "application/octet-stream"),
        "file2": ("modality1_patient0.nii", open("inference_modalities/modality1_patient0.nii", "rb"), "application/octet-stream"),
        "file3": ("modality2_patient0.nii", open("inference_modalities/modality2_patient0.nii", "rb"), "application/octet-stream"),
        "file4": ("modality3_patient0.nii", open("inference_modalities/modality3_patient0.nii", "rb"), "application/octet-stream"),
    }

    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        result = response.json()

        if "overlays" not in result or "report" not in result:
            raise ValueError("Response missing 'overlays' or 'report' key")
        if not isinstance(result["overlays"], dict) or not isinstance(result["report"], str):
            raise ValueError("'overlays' must be a dictionary and 'report' must be a string")

        save_overlays(result["overlays"], output_dir)
        print("Overlay images saved to:", output_dir)
        print("Received overlay keys:", list(result["overlays"].keys()))
        print("Medical Report:\n", result["report"])

    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {str(e)}")
    except ValueError as e:
        print(f"Validation or processing error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()