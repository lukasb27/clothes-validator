import argparse
import csv
import os
import pathlib
import tkinter as tk
from tkinter import ttk
from typing import List

from PIL import Image, ImageTk


class Labeller:
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path
        self._ensure_csv_exists_with_headers()
        self.files = self._get_files()
        self.image_label

    def _ensure_csv_exists_with_headers(self, label_path) -> None:
        if not os.path.exists(label_path):
            with open(label_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["image_path", "label"])

    def _get_files(self, data_path) -> List:
        path = pathlib.Path(data_path)
        return list(path.rglob("*.[jpeg jpg]*"))

    def _write_decision(self, image_path: str, decision: str, csv_path: str = CSV_PATH):
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_path, decision])
        self.update_image()

    def _update_image(self):
        if len(self.files) > 1:
            self.files.pop()
            img_path = self.files[-1]
            img = Image.open(img_path)
            img = img.resize((500, 500))
            image = ImageTk.PhotoImage(img)

            self.image_label.config(image=image)

            # Keep reference to avoid garbage collection
            self.image_label.image = image

    def label_dataset(self):
        root = tk.Tk()
        root.title("Image Classifier")
        root.bind(
            "<Right>", lambda x: self._write_decision(str(self.files[-1]), "liked")
        )
        root.bind(
            "<Left>", lambda x: self._write_decision(str(self.files[-1]), "disliked")
        )

        self.image_label = ttk.Label(root)
        self.image_label.pack()

        self._update_image()
        root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Dataset labeller")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/",
        help="The top level folder containing the dataset",
    )
    parser.add_argument("--label-path", type=str, default="labels.csv", help="")

    args = parser.parse_args()

    labeller = Labeller(args.data_path, args.label_path)
    labeller.label_dataset()


if __name__ == "__main__":
    pass
