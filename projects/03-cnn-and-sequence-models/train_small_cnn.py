"""Project 03, Part B: train a small CNN on the scikit-learn digits dataset with PyTorch.

The dataset has 1,797 images of handwritten digits (0 to 9), each 8x8 pixels.
It is bundled with scikit-learn, so no download is needed.
Runs on CPU in a few minutes.

Run:  python train_small_cnn.py
"""

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)  # keeps small CPU runs predictable

# Step 1: Load the images.
digits = load_digits()
images = digits.images.astype(np.float32)  # shape (1797, 8, 8), values 0 to 16
labels = digits.target.astype(np.int64)
print(f"Loaded {len(images)} images of size {images.shape[1]}x{images.shape[2]}")

# Step 2: Normalize pixels to 0 to 1, and add a channel dimension.
# CNNs in PyTorch expect shape (batch, channels, height, width).
images = images / 16.0
images = images[:, np.newaxis, :, :]  # shape (1797, 1, 8, 8)

# Step 3: Split into training and test sets (80/20, same class mix in each).
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=SEED, stratify=labels
)
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
print(f"Training images: {len(x_train)}, test images: {len(x_test)}")

# Step 4: Build a small CNN.
#   Conv -> ReLU -> Pool, twice, then Flatten -> Linear (the classifier).
#   8x8 input -> conv keeps 8x8 (padding=1) -> pool -> 4x4 -> conv -> pool -> 2x2
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),        # 8x8 -> 4x4
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),        # 4x4 -> 2x2
    nn.Flatten(),                       # 16 channels x 2 x 2 = 64 values
    nn.Linear(16 * 2 * 2, 10),          # 10 outputs, one per digit
)
print("\nModel:")
print(model)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def accuracy(inputs, targets):
    """Return the fraction of predictions that match the true labels."""
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)
    return (predictions == targets).float().mean().item()


# Step 5: Training loop. One epoch = one pass over all training images.
print("\nTraining:")
for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(len(x_train))
    running_loss = 0.0

    for start in range(0, len(x_train), BATCH_SIZE):
        batch_index = order[start:start + BATCH_SIZE]
        batch_x = x_train[batch_index]
        batch_y = y_train[batch_index]

        optimizer.zero_grad()                 # clear old gradients
        outputs = model(batch_x)              # forward propagation
        loss = loss_function(outputs, batch_y)
        loss.backward()                       # backward propagation
        optimizer.step()                      # update the weights

        running_loss += loss.item() * len(batch_index)

    average_loss = running_loss / len(x_train)
    train_acc = accuracy(x_train, y_train)
    print(f"Epoch {epoch:2d}/{EPOCHS}  loss {average_loss:.4f}  train accuracy {train_acc:.3f}")

# Step 6: Measure on the test set, which the model has never trained on.
test_acc = accuracy(x_test, y_test)
correct = round(test_acc * len(x_test))
print(f"\nTest accuracy: {test_acc:.3f} ({correct}/{len(x_test)} correct)")
