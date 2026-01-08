\# Medical AI Fracture Detection



A deep learning project for detecting bone fractures from wrist X-ray images using PyTorch.



\## Motivation

This project explores the application of convolutional neural networks to medical imaging, with a focus on fracture detection and model explainability.



\## Dataset

X-ray images organized into fractured and non-fractured classes.

Images are not included in this repository due to size and licensing constraints.



\## Approach

\- Transfer learning with a CNN backbone (PyTorch)

\- Binary classification: fractured vs non-fractured

\- Evaluation using confusion matrix, precision, sensitivity, and specificity

\- Grad-CAM used to visualize model attention regions



\## Results

The model demonstrates high sensitivity to fractures while exhibiting conservative behavior that leads to false positives.

Grad-CAM visualizations show the model focuses primarily on wrist bone regions.



\## Limitations

\- Limited dataset size

\- Conservative predictions reduce specificity

\- Model does not localize exact fracture lines



\## How to Run

```bash

python -m venv venv

source venv/bin/activate  # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

python src/train.py

python src/grad\_cam.py



