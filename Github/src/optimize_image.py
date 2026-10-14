import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2. Load trained ResNet50 model
model = models.resnet50(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load("models/ai_real_detector_resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully.")


# 3. Load one FAKE image
fake_folder = "Dataset/CIFAKE/test/FAKE"

image_name = os.listdir(fake_folder)[5]
image_path = os.path.join(fake_folder, image_name)

print("Selected image:", image_path)

# Same transform used during testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)


# 4. Predict original image
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

classes = ["FAKE", "REAL"]

print("Prediction:", classes[predicted_class])
print("Probabilities:", probabilities.cpu().numpy())


# 5. Make image trainable
optimized_image = input_tensor.clone().detach().requires_grad_(True)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Optimizer for image
optimizer = torch.optim.Adam(
    [optimized_image],
    lr=0.0005
)

# Target = REAL
target = torch.tensor([1]).to(device)


# 6. Optimization loop
num_steps = 100
lambda_reg = 10.0

for step in range(num_steps):

    optimizer.zero_grad()

    # Forward pass
    output = model(optimized_image)

    # Classification loss
    classification_loss = nn.CrossEntropyLoss()(
        output,
        target
    )

    # Regularization loss (MSE)
    regularization_loss = torch.mean(
        (optimized_image - input_tensor) ** 2
    )

    # Total loss
    total_loss = (
        classification_loss
        + lambda_reg * regularization_loss
    )

    # Backpropagation
    total_loss.backward()

    optimizer.step()

    # Clamp values
    optimized_image.data = torch.clamp(
        optimized_image.data,
        -3,
        3
    )

    # Print progress
    if step % 20 == 0:

        probs = torch.softmax(output, dim=1)

        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()

        print(f"Step {step}")
        print(f"Fake Probability: {fake_prob:.4f}")
        print(f"Real Probability: {real_prob:.4f}")
        print(f"Classification Loss: {classification_loss.item():.8f}")
        print(f"Regularization Loss: {regularization_loss.item():.8f}")
        print("-" * 40)



# 7. Convert tensor back to image
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return tensor * std + mean

original_img = denormalize(input_tensor).detach().cpu().squeeze(0)
optimized_img = denormalize(optimized_image).detach().cpu().squeeze(0)

original_img = torch.clamp(original_img, 0, 1)
optimized_img = torch.clamp(optimized_img, 0, 1)

original_img_np = original_img.permute(1, 2, 0).numpy()
optimized_img_np = optimized_img.permute(1, 2, 0).numpy()


# 8. Save and show images
os.makedirs("results", exist_ok=True)

plt.imsave("results/original_ai_image.png", original_img_np)
plt.imsave("results/optimized_ai_to_real.png", optimized_img_np)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_img_np)
plt.title("Original AI Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(optimized_img_np)
plt.title("Optimized Image")
plt.axis("off")

plt.tight_layout()
plt.savefig("results/comparison.png")
plt.show()

print("Saved images:")
print("results/original_ai_image.png")
print("results/optimized_ai_to_real.png")
print("results/comparison.png")