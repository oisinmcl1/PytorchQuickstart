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
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
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
            nn.Dropout(0.2) , # Dropout layer for regularization
            nn.Linear(512, 512), # Hidden layer converting 512 inputs to 512 outputs
            nn.ReLU(), # ReLU activation function
            nn.Dropout(0.2), # Dropout layer for regularization
            nn.Linear(512, 10) # Output layer converting 512 inputs to 10 outputs (one for each class)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# ==========

# ===== Training and  Test =====

def train(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        # Print loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # Accumulate loss
            test_loss += loss_fn(pred, y).item()
            # Count correct predictions
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Compute average loss and accuracy
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# ==========

# ===== Run Model =====

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimiser)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Save model state
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# ==========
