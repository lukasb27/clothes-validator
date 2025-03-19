from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torch
from torch import optim
from torch import nn
from torchvision import transforms
from .custom_image_dataset import CustomImageDataset
from .neural_network import NeuralNetwork 

def get_custom_dataset(transform=None) -> CustomImageDataset:
    return CustomImageDataset("labels.csv", transform)

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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), 'model.pth')

def do_something():
    transform = v2.Compose([
        transforms.Resize((28,28))
    ])
    training_dataset = get_custom_dataset(transform=transform)
    training_dataloader = get_data_loader(training_dataset)
    train_features, train_labels = next(iter(training_dataloader))
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    model = NeuralNetwork().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, training_dataloader, criterion, optimiser, device, num_epochs=10)

if __name__ == "__main__":
    pass