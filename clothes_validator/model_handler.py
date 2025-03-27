import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2

from .custom_image_dataset import CustomImageDataset
from .neural_network import NeuralNetwork


def get_custom_dataset(label_file="labels.csv", transform: transforms.Compose =None) -> CustomImageDataset:
    """Get custom dataset object.

    :param label_file: Path to the csv label, defaults to "labels.csv".
    :type label_file: str, optional
    :param transform: The transforms to perform on the data, defaults to None
    :type transform: transforms.Compose, optional - https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html
    :return: _description_
    :rtype: CustomImageDataset - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    return CustomImageDataset(label_file, transform)


def get_data_loader(dataset: CustomImageDataset) -> DataLoader:
    """Get data loader object from a dataset.
    
    :param dataset: A dataset object.
    :type dataset: CustomImageDataset
    :return: DataLoader, a dataset iterable.
    :rtype: DataLoader - https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    return DataLoader(dataset, batch_size=64, shuffle=True)


def train_model(model: NeuralNetwork, dataloader: DataLoader, optimiser: optim.Adam, device: torch.device, num_epochs=10, criterion = nn.CrossEntropyLoss()):

    # Set model to train mode.
    model.train()

    # Epochs are one whole pass through the dataset. The model will take in all the data at once.
    # Multiple Epochs are set, so the model runs through training data more than once,
    # this increases the accuracy of the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Get the batch of data from the dataloder. Remember that dataloader is an iterable object based on the dataset.
        # The batch size is defined when the dataloader is instantiated, but defaults to 64 items.
        for batch in dataloader:
            # Type hint to help me remember whats going on below.
            inputs: Tensor
            labels: Tensor

            # Unpack the batch tuple into input tensors and labels tensors into 1D Arrays.
            inputs, labels = batch

            # Send the tensors to the correct device, based on the device param.
            inputs, labels = inputs.to(device), labels.to(device)

            # Reset the gradients from the previous run so they dont poison this run.
            optimiser.zero_grad()

            # Pass the inputs through the models layers (Forward pass)
            outputs = model(inputs) 

            loss = criterion(outputs, labels)  
            loss.backward()  
            optimiser.step()  

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}"
        )
    return model


def validate_model(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            print(predicted, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Model handler. Responsible for either training or loading a model."
    )
    parser.add_argument(
        "--training-label-path",
        type=str,
        default="labels.csv",
        help="Path to the training label csv",
    )
    parser.add_argument(
        "--validation-label-path",
        type=str,
        default="validation_labels.csv",
        help="Path to the validation label csv",
    )
    parser.add_argument(
        "--model-file-path",
        type=str,
        default="model.pth",
        help="Path to the model pth file",
    )
    args = parser.parse_args()

    transform = v2.Compose([transforms.Resize((28, 28))])
    training_dataset = get_custom_dataset(args.training_label_path, transform)
    training_dataloader = get_data_loader(training_dataset)

    validation_dataset = get_custom_dataset(args.validation_label_path, transform)
    validation_dataloader = get_data_loader(validation_dataset)

    device = torch.device("mps")

    model = NeuralNetwork().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists(args.model_file_path):
        model.load_state_dict(torch.load(args.model_file_path, weights_only=True))
    else:
        model = train_model(
            model, training_dataloader, optimiser, device, num_epochs=10
        )
        torch.save(model.state_dict(), args.model_file_path)

    accuracy = validate_model(model, validation_dataloader, device)
    print(f"Accuracy is {accuracy}%")


if __name__ == "__main__":
    pass
