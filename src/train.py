import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from data_loader import FractureDataset
from model import get_model

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets
train_ds = FractureDataset(
    root_dir="data/raw/x-rayData/train",
    transform=transform
)

val_ds = FractureDataset(
    root_dir="data/raw/x-rayData/val",
    transform=transform
)

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16)

# Model
model = get_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training
epochs = 3

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1} Training"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(val_dl, desc="Validation"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print("\n==============================")
    print(f"Epoch {epoch+1} Results")
    print("==============================")
    print("Confusion Matrix:")
    print(cm)
    print(f"Precision:   {precision:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print("==============================\n")

# Save model
torch.save(model.state_dict(), "outputs/fracture_model.pt")
print("Model saved to outputs/fracture_model.pt")

