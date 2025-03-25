import pytest
import csv
import os

import torch
from clothes_validator.model_handler import get_custom_dataset, get_data_loader
from clothes_validator.custom_image_dataset import CustomImageDataset
from conftest import CSV_FILE

class MockCustomImage():
    def __init__(self, annotations_file, transform=None, target_transform=None):
        pass
    def __len__(self):
        return 1
    def __get__item(self, idx):
        return torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))
    
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