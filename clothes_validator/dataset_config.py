import argparse
import os
import shutil

import kaggle


def download_dataset(dataset: str, path: str) -> None:
    path = f"{path}/{dataset.replace("/", "+")}"
    if os.path.exists(path):
        print(f"Data exists at {path}, please provide an empty path.")
    else:
        print("Downloading dataset")
        kaggle.api.dataset_download_files(dataset, path=path, unzip=True)
        print("Download complete!")


def remove_dataset(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        print("Path doesnt exist")


def main():
    parser = argparse.ArgumentParser(description="Dataset downloader/remover")
    parser.add_argument(
        "--action",
        type=str,
        choices=["download", "delete"],
        required=True,
        help="The action to take, options are: download or delete",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="The path to download the dataset to or to delete, defaults to ./data",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sunnykusawa/tshirts",
        help="The full kaggle dataset name to download, defaults to sunnykusawa/tshirts",
    )

    args = parser.parse_args()

    actions = {
        "download": download_dataset(args.dataset_name, args.data_path),
        "delete": remove_dataset(args.data_path),
    }

    actions[args.action]


if __name__ == "__main__":
    pass
