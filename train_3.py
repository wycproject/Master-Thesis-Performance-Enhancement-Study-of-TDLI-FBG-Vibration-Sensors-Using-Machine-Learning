import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np
import os
import psutil

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
train_dir = r"E:\Dataset3\TrainDataset2\train"
test_dir = r"E:\Dataset3\TrainDataset2\test"

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((420, 560)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Classes
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}, Total Classes: {num_classes}")

# Define DNN (without built-in dropout)
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc5 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Parameters
input_size = 420 * 560
hidden_size = 128
num_epochs = 100
learning_rate = 0.001
patience = 7  # Early stopping patience

# Initialize model, loss, optimizer
model = DNNModel(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Metrics
batch_metrics = []
epoch_metrics = []

# Early stopping
best_val_loss = np.inf
epochs_no_improve = 0

print("\n🔄 Training started...")
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        # Apply dropout externally
        outputs = F.dropout(outputs, p=0.3, training=True)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

        # Validation step per batch
        model.eval()
        correct_val = 0
        total_val = 0
        validation_loss = 0.0

        with torch.no_grad():
            for val_images, val_labels in test_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_outputs = F.dropout(val_outputs, p=0.3, training=False)
                val_loss = criterion(val_outputs, val_labels)
                validation_loss += val_loss.item()

                _, val_predicted = torch.max(val_outputs, 1)
                correct_val += (val_predicted == val_labels).sum().item()
                total_val += val_labels.size(0)

        batch_validation_accuracy = 100 * correct_val / total_val
        avg_validation_loss = validation_loss / len(test_loader)

        # Store batch metrics
        batch_metrics.append([
            epoch+1, batch_idx+1, loss.item(), correct_train / total_train, avg_validation_loss, batch_validation_accuracy
        ])

    # Epoch summary
    epoch_training_accuracy = 100 * correct_train / total_train
    epoch_validation_accuracy = batch_validation_accuracy
    epoch_time = time.time() - epoch_start_time

    # Store epoch metrics
    epoch_metrics.append([
        epoch+1, epoch_training_accuracy, epoch_validation_accuracy, epoch_time
    ])

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Acc: {epoch_training_accuracy:.2f}%, Val Acc: {epoch_validation_accuracy:.2f}%, Val Loss: {avg_validation_loss:.4f}")

    # Early stopping check
    if avg_validation_loss < best_val_loss:
        best_val_loss = avg_validation_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_dnn_model_run3_2.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("\nEarly stopping triggered.")
            break

# Save metrics
pd.DataFrame(batch_metrics, columns=[
    "Epoch", "Batch", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"
]).to_csv("batch_metrics_2.csv", index=False)

pd.DataFrame(epoch_metrics, columns=[
    "Epoch", "Training Accuracy", "Validation Accuracy", "Epoch Time"
]).to_csv("epoch_metrics_2.csv", index=False)

print(f"\nModel saved to: best_dnn_model_run3_2.pth")
print("\nTraining complete! Training stats saved in CSV.")
