from groq import Groq
import os

def generate_report(regions, patient_id="anonymous"):
    try:
        
        api_key= os.getenv("GROQ_API_KEY")
        if not api_key:
          raise ValueError("Missing GROQ_API_KEY! Please set it in your environment or .env file.")
        client = Groq(api_key=api_key)

        num_lesions = len(regions)
        total_volume = sum(region["volume_cm3"] for region in regions.values())
        lesion_str = "\n".join([
            f"  - Lesion {id}: Volume {region['volume_cm3']:.3f} cm³, Location: {region['location']}"
            for id, region in regions.items()
        ])

        prompt = f"""
        You are a neurologist writing a professional medical report for a patient with multiple sclerosis (MS). Using only the provided imaging data, generate a report with sections for Patient Information, Imaging Details, Findings, Impression, and Recommendations. The tone should be formal, concise, and professional, suitable for a clinical setting. Do not invent data beyond what is provided, including clinical history or patient demographics (e.g., age, sex), as these are unavailable. Ensure the report is HIPAA-compliant (no real patient identifiers). Use the JHU ICBM-DTI-81 and Harvard-Oxford atlases for anatomical reference.

        **Data**:
        - Patient: Anonymous, ID {patient_id}
        - Imaging: MRI brain, T2-FLAIR-MPRAGE-PD, 1 mm isotropic resolution, acquired July 1, 2025
        - Findings:
          - Number of lesions: {num_lesions}
          - Lesion details:
        {lesion_str}
          - Total lesion volume: {total_volume:.3f} cm³

        Format the report as plain text with clear section headers. Group lesion locations by anatomical region (e.g., cortical, white matter, subcortical, juxtacortical) and highlight clinically significant findings (e.g., large lesions or critical locations like the corpus callosum). Provide recommendations based solely on the imaging findings.
        """

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            stream=False,
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        raise ValueError(f"Failed to generate report: {str(e)}")