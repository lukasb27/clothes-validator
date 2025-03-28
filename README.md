# Clothes Validator

This project can:
- Download training datasets from Kaggle.
- Label the dataset with user input. 
- Train the dataset.
- Validate the dataset.

## Downloading/removing dataset

To download/remove a kaggle dataset use the dataset command.

```
> poetry run dataset -h

usage: dataset [-h] --action {download,delete} [--data-path DATA_PATH] [--dataset-name DATASET_NAME]

Dataset downloader/remover

options:
  -h, --help            show this help message and exit
  --action {download,delete}
                        The action to take, options are: download or delete
  --data-path DATA_PATH
                        The path to download the dataset to or to delete, defaults to "./data"
  --dataset-name DATASET_NAME
                        The full kaggle dataset name to download, defaults to "sunnykusawa/tshirts"
```

## Labelling dataset

To label a dataset, run the below command. Use the right arrow to label the image as "liked" and the left arrow to label the image as "disliked". This will create a csv file at the label path.

```
> poetry run label_dataset -h

usage: label_dataset [-h] [--data-path DATA_PATH] [--label-path LABEL_PATH]

Dataset labeller

options:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        The top level folder containing the dataset
  --label-path LABEL_PATH
                        The path to store the label to
```

## Training the model

If you want to train a model, use the create-model command. If you want to just get a model object you can load a model from a model save dict path by passing in the --model-file-path.
The first time you run this the model will save it to the model file path, which defaults to model.pth.

```
> poetry run create_model -h

usage: create_model [-h] [--training-label-path TRAINING_LABEL_PATH] [--validation-label-path VALIDATION_LABEL_PATH] [--model-file-path MODEL_FILE_PATH]

Model handler. Responsible for either training, loading or validating a model.

options:
  -h, --help            show this help message and exit
  --training-label-path TRAINING_LABEL_PATH
                        Path to the training label csv
  --validation-label-path VALIDATION_LABEL_PATH
                        Path to the validation label csv
  --model-file-path MODEL_FILE_PATH
                        Path to the model pth file
```