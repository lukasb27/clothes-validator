from clothes_validator.model_handler import get_custom_dataset
from clothes_validator.custom_image_dataset import CustomImageDataset
from torch import Tensor
from conftest import CSV_FILE

def test_custom_dataset(create_labels):
    custom_dataset = CustomImageDataset(CSV_FILE)

    # # Assert the dunders are returning as expected
    assert len(custom_dataset)
    assert custom_dataset[0]

    # Assert the custom data is returning Tensors
    assert type(custom_dataset[0][0]) is Tensor
    assert type(custom_dataset[0][1]) is Tensor


