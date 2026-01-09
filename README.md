# Medical AI Fracture Detection with Grad-CAM

## Overview
This project implements a deep learning pipeline for **binary fracture detection** using X-ray images. A Convolutional Neural Network (CNN) is trained to classify images as **Fractured** or **Not Fractured**, with **Grad-CAM visualizations** used to interpret model predictions.

The project was inspired by a personal wrist injury and aims to explore **medical imaging, model interpretability, and real-world machine learning challenges**.

---

## Dataset
- X-ray image dataset
- Two classes:
  - Fractured
  - Not Fractured
- Folder-based labeling (no CSV files)
- Directory structure:
  - `train/`
  - `val/`
- Image format: `.jpg`

---

## Model Architecture
- Convolutional Neural Network (CNN)
- Implemented in PyTorch
- Binary classification output
- Loss function: Cross-Entropy Loss
- Optimizer: Adam

---

## Training & Evaluation
The model was trained over multiple epochs and evaluated on a validation set.

### Example Metrics
- Precision: ~0.62  
- Sensitivity (Recall): ~0.81  
- Specificity: ~0.26  

These results reflect common **medical AI trade-offs**, where higher sensitivity is often prioritized to reduce missed fractures.

---

## Model Interpretability (Grad-CAM)
Grad-CAM is used to visualize which regions of an X-ray most influenced the model’s predictions. This helps improve transparency and trust in the model’s decision-making process.

### Example Outputs
<img width="1221" height="438" alt="Screenshot 2026-01-07 232440" src="https://github.com/user-attachments/assets/2f63fd3f-4490-4972-92e6-6d4ec51a7a8a" />

## How to Run
### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Train model
```bash
python src/train.py
```
### 4. Generate Grad-CAM visualizations
```bash
python src/grad_cam.py
```
## Project Structure
```text
medical-ai-fracture-detection/
├── data/
│   └── raw/
│       └── x-rayData/
│           ├── train/
│           └── val/
├── results/
├── src/
│   ├── train.py
│   ├── data_loader.py
│   ├── model.py
│   └── grad_cam.py
├── requirements.txt
├── README.md
└── .gitignore
```
## Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy
- Grad-CAM
- Git / GitHub

## Future Work
- Improve class balance
- Transfer learning with pretrained medical models
- Add MRI-based ligament/tendon detection
- Multi-label classification
- ROC-AUC and clinical-grade evaluation metrics

## Disclaimer

This project is for educational and research purposes only and is not intended for clinical use.
