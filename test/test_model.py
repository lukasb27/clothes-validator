import pytest
import csv
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch import Tensor, nn, optim

from clothes_validator.model_handler import get_custom_dataset, get_data_loader, train_model
from clothes_validator.custom_image_dataset import CustomImageDataset
from conftest import CSV_FILE

from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Define the layers in the Neural Network
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
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

    model = NeuralNetwork().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(model, dataloader, optimiser, torch.device("mps"), num_epochs=1)

    assert type(model) is NeuralNetwork

    # Test that the model created from train_model is able to make predictions.
    model.eval()

    test_inputs, _ = next(iter(dataloader))
    test_inputs = test_inputs.to(device)

    with torch.no_grad():
        outputs = model(test_inputs)
        _, predicted = torch.max(outputs, 1)
    
    assert torch.is_tensor(predicted)
 
