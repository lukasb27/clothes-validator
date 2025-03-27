import pytest
import csv
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch import Tensor, nn, optim

from clothes_validator.model_handler import get_custom_dataset, get_data_loader, train_model, validate_model
from clothes_validator.custom_image_dataset import CustomImageDataset
from conftest import CSV_FILE

from torch import nn

class MockNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Define the layers in the Neural Network
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28 * 3, 512),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class MockCustomImage(Dataset):
    def __init__(self, annotations_file=None, transform=None, target_transform=None):
        pass
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        # This needs to be 3, 28, 28 as the 3 represents RGB channels. 1 channel ie (1, 28, 28) would represent grayscale.
        return torch.randn(3, 28, 28), torch.tensor(1, dtype=torch.long)

class MockFixedModel(nn.Module):
    """ To test the validation algorithm we need to fix the models outputs """
    def __init__(self, outputs):
        # Ensure that we initialise the nn.Module so we can only overwrite the methods we need to fix.
        super(MockFixedModel, self).__init__() 
        # Pass in the outputs we want the 'model' to return.
        self.fixed_outputs = outputs
        self.counter = 0
    
    def forward(self, x):
        batch_size = x.shape[0]
        start = self.counter*batch_size
        forward = self.fixed_outputs[start:start+batch_size]
        self.counter += 1
        return forward


class MockDataLoader:
    """ To test the validation algorithm we need to fix the inputs/labels on the dataloader."""
    def __init__(self, inputs, labels, batch_size = 2):
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
    
    def __iter__(self):
        for i in range(0, len(self.inputs), self.batch_size):
            yield self.inputs[i : i + self.batch_size], self.labels[i : i + self.batch_size]
    
    def __len__(self):
        return len(self.inputs) // self.batch_size


@pytest.fixture
def dataloader():
    dataset = MockCustomImage()
    yield DataLoader(dataset)

def test_get_dataset(create_labels):
    custom_dataset = get_custom_dataset(CSV_FILE)

    # Assert it returns something
    assert custom_dataset

    # Assert the return type is correct 
    assert type(custom_dataset) is CustomImageDataset




def test_get_data_loder():
    test_dataloader = get_data_loader(MockCustomImage("test.csv"))
    
    assert test_dataloader
    assert type(test_dataloader) is torch.utils.data.DataLoader


def test_training_model(dataloader):
    device = torch.device("mps")

    model = MockNeuralNetwork().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, dataloader, optimiser, torch.device("mps"), num_epochs=1)

    assert type(model) is MockNeuralNetwork
    assert next(model.parameters()).is_mps

    # Test that the model created from train_model is able to make predictions.
    model.eval()

    test_inputs, _ = next(iter(dataloader))
    test_inputs = test_inputs.to(device)

    with torch.no_grad():
        outputs = model(test_inputs)
        _, predicted = torch.max(outputs, 1)
    
    assert torch.is_tensor(predicted)


def test_validation_works():
    device = torch.device("mps")

    test_inputs =  torch.randn(4, 1, 28, 28)
    test_labels = torch.tensor([0, 1, 1, 0])

    # The algo will return the index of the highest number (0/1) as its prediction. 
    # Ensure to send the outputs to the device too.
    test_outputs = torch.tensor([
        [0.1, 0.0], #Predict 0
        [0.2, 0.8], #Predict 1
        [0.4, 0.6], #Predict 1
        [0.3, 0.7], #Predict 0
    ]).to(device)

    # Instatiate our mock objects with the fixed outputs/inputs/labels.
    model = MockFixedModel(test_outputs).to(device)
    dataloader = MockDataLoader(test_inputs, test_labels)
    
    # Get the accuracy of the model. 
    accuracy = validate_model(model, dataloader, device)

    assert accuracy == 75.0
