import pytest
import os 
import csv

CSV_FILE = "test_label.csv"

@pytest.fixture
def create_labels():
    with open(CSV_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "label"])
        writer.writerow(["test/test_images/test_t_shirt.jpeg", "disliked"])
    yield
    os.remove(CSV_FILE)