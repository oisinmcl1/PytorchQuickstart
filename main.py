import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ===== Prepare dataset =====

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# ==========

# ===== Model definition =====

# Get cpu or gpu device for training.
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"\nUsing {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Flatten 28x28 images to a 784 vector
        self.flatten = nn.Flatten()

        # Neural network architecture
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # Input layer converting 784 inputs to 512 outputs
            nn.ReLU(), # ReLU activation function
            nn.Linear(512, 512), # Hidden layer converting 512 inputs to 512 outputs
            nn.ReLU(), # ReLU activation function
            nn.Linear(512, 10) # Output layer converting 512 inputs to 10 outputs (one for each class)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# ==========
