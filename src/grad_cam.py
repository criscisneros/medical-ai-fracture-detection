import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model import get_model

# --------------------
# Config
# --------------------
IMAGE_PATH = "data/raw/x-rayData/val/fractured"  # folder
MODEL_PATH = "outputs/fracture_model.pt"
TARGET_CLASS = 1  # 1 = fractured

# --------------------
# Load model
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --------------------
# Pick one image
# --------------------
import os
img_name = os.listdir(IMAGE_PATH)[0]
img_path = os.path.join(IMAGE_PATH, img_name)

original_img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

input_tensor = transform(original_img).unsqueeze(0).to(device)

# --------------------
# Grad-CAM Hook
# --------------------
gradients = []
activations = []

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

def forward_hook(module, input, output):
    activations.append(output)

target_layer = model.layer4[-1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# --------------------
# Forward + backward
# --------------------
output = model(input_tensor)
model.zero_grad()
output[0, TARGET_CLASS].backward()

# --------------------
# Compute CAM
# --------------------
grads = gradients[0].cpu().data.numpy()
acts = activations[0].cpu().data.numpy()

weights = np.mean(grads, axis=(2, 3))[0]
cam = np.zeros(acts.shape[2:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * acts[0, i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam / cam.max()

# --------------------
# Overlay
# --------------------
img_np = np.array(original_img.resize((224, 224)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

# --------------------
# Show
# --------------------
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original X-ray")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM")
plt.imshow(cam, cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.show()