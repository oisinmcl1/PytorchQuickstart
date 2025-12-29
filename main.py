import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

# Class labels for Fashion MNIST
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ===== VISUALIZATION 1: Sample Images from Dataset =====

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample Fashion MNIST Images', fontsize=16)

for i, ax in enumerate(axes.flat):
    img, label = training_data[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(classes[label])
    ax.axis('off')

plt.tight_layout()
plt.show()

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
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10) # 10 output classes (one for each clothing category)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# ==========

# ===== Training and Test Functions =====

def train(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()
    epoch_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        epoch_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        # Print loss every 100 batches
        if batch % 100 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

    return epoch_loss / len(dataloader)


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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct

# ==========

# ===== Run Model =====

loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

# Initialize tracking lists
train_losses = []
test_losses = []
test_accuracies = []

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimiser)
    test_loss, accuracy = test(test_dataloader, model, loss_fn)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)

print("Done!")

# Save model state
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# ==========

# ===== VISUALIZATION 2: Training History =====

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, epochs + 1), train_losses, 'b-', label='Training Loss', marker='o')
ax1.plot(range(1, epochs + 1), test_losses, 'r-', label='Test Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, epochs + 1), [acc * 100 for acc in test_accuracies], 'g-', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Test Accuracy Over Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========

# ===== VISUALIZATION 3: Confusion Matrix =====

# Get predictions for entire test set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_dataloader:
        X = X.to(device)
        pred = model(X)
        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ==========

# ===== VISUALIZATION 4: Prediction Examples =====

# Show some predictions
model.eval()
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Model Predictions vs Actual Labels', fontsize=16)

with torch.no_grad():
    for i, ax in enumerate(axes.flat):
        img, true_label = test_data[i]
        img_batch = img.unsqueeze(0).to(device)
        pred = model(img_batch)
        pred_label = pred.argmax(1).item()

        ax.imshow(img.squeeze(), cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'Pred: {classes[pred_label]}\nTrue: {classes[true_label]}',
                     color=color, fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.show()

# ==========

# ===== VISUALIZATION 5: Per-Class Accuracy =====

# Calculate per-class accuracy
class_correct = [0] * 10
class_total = [0] * 10

model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        predictions = pred.argmax(1)

        for label, prediction in zip(y, predictions):
            if label == prediction:
                class_correct[label] += 1
            class_total[label] += 1

class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(10)]

plt.figure(figsize=(12, 6))
bars = plt.bar(classes, class_accuracies, color='skyblue', edgecolor='navy')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Per-Class Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{acc:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ==========

print("\n=== Final Results ===")
print(f"Final Test Accuracy: {test_accuracies[-1] * 100:.2f}%")
print(f"Final Test Loss: {test_losses[-1]:.4f}")
print("\nPer-class accuracies:")

for class_name, acc in zip(classes, class_accuracies):
    print(f"  {class_name:15s}: {acc:5.2f}%")
