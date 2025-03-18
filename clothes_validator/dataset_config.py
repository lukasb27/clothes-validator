import kaggle
import os
import shutil

def download_dataset(dataset: str="sunnykusawa/tshirts", path = "./data") -> None:
    path = f"{path}/{dataset.replace("/", "+")}"
    if os.path.exists(path):
        print(f"Data exists at {path}, please provide an empty path.")
    else:
        print("Downloading dataset")
        kaggle.api.dataset_download_files(dataset, path=path, unzip=True)
        print("Download complete!")
    
def remove_dataset(path = "./data") -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        print("Path doesnt exist")
    

if __name__ == "__main__":
    pass