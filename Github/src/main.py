import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


# 1. Paths
dataset_path = "Dataset/CIFAKE"  

train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")


# 2. Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



# 3. Image transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



# 4. Load dataset
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

print("Classes:", train_data.classes)
print("Class mapping:", train_data.class_to_idx)
print("Train size:", len(train_data))
print("Test size:", len(test_data))


# 5. DataLoaders
batch_size = 32

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# 6. Model: ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block (better learning)
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer for 2 classes (FAKE, REAL)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Move model to GPU
model = model.to(device)


# 7. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

# 8. Training
num_epochs = 3

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        loop.set_postfix(
            loss=loss.item(),
            acc=train_acc
        )

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Training Loss: {avg_loss:.4f}")
    print(f"Training Accuracy: {train_acc:.2f}%")



# 9. Testing
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")


# 10. Save model
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/ai_real_detector_resnet50.pth")

print("Model saved at: models/ai_real_detector_resnet50.pth")


# 11. Evaluation
model.eval()

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in tqdm(test_loader, desc="Evaluation"):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Final Accuracy
test_accuracy = 100 * correct / total

print(f"Test Accuracy: {test_accuracy:.2f}%")

# 12. Confusion Matrix + Classification Report
# 12. Confusion Matrix + Classification Report
class_names = train_data.classes

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()