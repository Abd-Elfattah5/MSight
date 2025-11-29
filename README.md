# Auto MRI Segmentation for MS Diagnosis

**Top 4 Graduation Project**

---

## Overview
This project presents an automated system for MRI lesion segmentation aimed at supporting Multiple Sclerosis (MS) diagnosis.  
It combines advanced deep learning architectures with efficient deployment and reporting pipelines to accelerate analysis and improve diagnostic accuracy.  
The system was trained and evaluated using the **ISBI2015 dataset**.

---

## Key Features

- **Attention‑based U‑Net Segmentation**  
  Built a U‑Net with attention mechanisms for MRI lesion segmentation.  
  Achieved a **69% Dice score** while reducing analysis time by **90%** (from 20:50 min → 5:10 sec).

- **Optimized Preprocessing Pipeline**  
  Implemented a fully optimized pipeline including normalization, resampling to MNI space, and stacking scans into tensors.  
  Standardized MRI inputs and improved overall model robustness.

- **Scalable Model Deployment**  
  Deployed the model using LightAPI for scalable training and efficient experimentation.

- **Lesion Quantification & LLM‑Based Reporting**  
  Integrated lesion quantification and anatomical mapping with the JHU atlas.  
  Coupled results with an LLM‑based reporting system to generate detailed clinical summaries.

- **End‑to‑End Web Application**  
  Developed a fully functional web application with backend and database integration.  
  Supports real‑time MRI upload, lesion segmentation, and automated report generation.

---

---

## Technical Stack

- **Frameworks**
  - PyTorch
  - MySQL
  - .Net

- **Deployment & Tools**
  - LightAPI

- **Libraries**
  - NiLearn\Nibabel
  - Pandas
  - NumPy
  - scikit-learn
  - Matplotlib
  - SciPy
  - CV2

- **Environment**
  - litserve==0.2.13
  - uvicorn==0.30.1
  - torch==2.1.0
  - torchvision==0.16.0
  - nibabel==5.2.0
  - numpy==1.24.3
  - matplotlib==3.7.1
  - scipy==1.11.4
  - pandas==2.2.0
  - torchio==0.19.6
  - opencv-python==4.7.0.72
  - pyarrow==17.0.0
  - plotly==6.2.0
  - nilearn==0.12.0
  - groq==0.30.0

---

## Project UI
You can use the project here:
[MSight Project UI](https://linktr.ee/_MSight)
### WARNING: the AI server might be down at random times, so although you can surf the website, you might not be able to infer data into insights.
