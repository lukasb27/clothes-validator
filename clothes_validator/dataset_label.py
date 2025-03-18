import tkinter as tk
from tkinter import ttk
from typing import List
from PIL import ImageTk, Image
import pathlib
import csv
import os

def get_files() -> List:
    path = pathlib.Path(f"./data/")
    return list(path.rglob("*.jpg"))

FILES = get_files()
CSV_PATH = "labels.csv"

def ensure_csv_exists_with_headers(csv_path = CSV_PATH) -> None:
    if not os.path.exists(CSV_PATH):
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image_path", "label"])

def label_dataset():
    def decision(image_path: str, decision: str, csv_path: str = CSV_PATH):
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_path, decision])
        update_image()
        os.remove(image_path)

    def update_image():
        if FILES:
            FILES.pop()
            img_path = FILES[-1]
            img = Image.open(img_path)
            img = img.resize((500, 500))  # Resize if necessary
            image = ImageTk.PhotoImage(img)

            image_label.config(image=image)
            image_label.image = image  # Keep reference to avoid garbage collection

    root = tk.Tk() 
    root.title("Image Classifier")
    root.bind("<Right>", lambda x: decision(str(FILES[-1]), "liked"))
    root.bind("<Left>", lambda x: decision(str(FILES[-1]), "disliked"))

        
    image_label = ttk.Label(root)
    image_label.pack()

    ensure_csv_exists_with_headers()
    update_image() 
    root.mainloop()
        