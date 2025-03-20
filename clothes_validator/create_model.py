import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2

from .custom_image_dataset import CustomImageDataset
from .neural_network import NeuralNetwork


def get_custom_dataset(transform=None, label="labels.csv") -> CustomImageDataset:
    return CustomImageDataset(label, transform)


def get_data_loader(dataset: CustomImageDataset) -> DataLoader:
    return DataLoader(dataset, batch_size=64, shuffle=True)


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = inputs.float()

            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}"
        )
    torch.save(model.state_dict(), "model.pth")
    return model


def validate_model(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
    return accuracy


def main():
    transform = v2.Compose([transforms.Resize((28, 28))])
    training_dataset = get_custom_dataset(transform=transform)
    training_dataloader = get_data_loader(training_dataset)

    validation_dataset = get_custom_dataset(transform, "validation_labels.csv")
    validation_dataloader = get_data_loader(validation_dataset)

    train_features, train_labels = next(iter(training_dataloader))
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    criterion = nn.CrossEntropyLoss()
    model = NeuralNetwork().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", weights_only=True))
    else:
        model = train_model(
            model, training_dataloader, criterion, optimiser, device, num_epochs=10
        )

    accuracy = validate_model(model, validation_dataloader, device)
    print(f"Accuracy is {accuracy}%")


if __name__ == "__main__":
    pass
