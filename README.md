# YoloV5 data preprocessing pipeline


This repo preprocesses video data from Tator.io to training in the Yolo network.


## Steps


1. Ingest media from Tator.io using API token.
2. Get the localizations associated with the media and divide them into train/validation/test sets.
3. Convert the localizations to COCO format and download the images into the local disk.
4. Convert the COCO format to Yolo and create labels for localizations.
5. Train the YOLO model.
6. Create the strategy.yaml to be used for detection.



### Prerequisites


- Python 3.7 or later
- Tator 0.10.25 0r later
- YoloV5 7.0 or later




## File Structure


The repository contains the following files and directories:


- `pipeline/`: Pipeline Python scripts
   - `saveLocalizations.py`: Ingest media from Tator.io using API token. Get the localizations associated with the media and divide them into train/validation/test sets. This script creates three pickle files named train_locs.pkl, val_locs.pkl, test_locs.pkl. Optionally, creates a CSV file named negative_ids.csv with the media ids of the negative localizations. Additionally, it creates a yaml file that contains all the path information of all the files created above.
   - `locsImgCoco.py`: Using the yaml created in the previous step, convert the localizations to COCO format and download the images to the local disk. This script downloads and stores the images locally into the path passed as the arguments. Also, converts the localizations into COCO format and stores it in a JSON file named coco_train, coco_val and coco_test.
   - `coco_to_yolo.py`: Convert the COCO format to Yolo and create labels for localizations. This script creates a yaml with a user-defined name and location ready for use in the yolo network. Additionally, this script creates labels for all the positive localizations in their respective file locations.
   - `create_strategy_yaml.py`: Ingest model and video metadata and return a yaml file to be used for detection.
  


## Usage


To run the preprocessing pipeline, follow these steps:


1. run `saveLocalization.py <api-token> <projectId> <sectionId> <testSize=0.4> <positives=1> <negatives=-1> <negative_set_size=0.1>`
   - api-token, projectId and sectionId are the attributes for tator API.
   - positives and negatives are the designated int value for the corrected field in tator for positive and negative samples, respectively.
   - testSize and negative_set_size are the float values or percentage of the train/val/test split and positive/negative sample sizes, respectively.
2. run `locsImgCoco.py <path-to-locsImgCoco.yaml-file>`
   - args_yaml is the dictionary created by reading the yaml file passed as an argument. The dictionary contains the below:
      - api-token is the attribute for tator api.
      - train-localization-path, val-localization-path, and test-localization-path are file paths to train/val/test localization pickle files created in the previous step.
      - coco-path is the file path to store the COCO formatted JSON file.
      - images-path is the folder to download and store the images locally in three subfolders called train, val and test.
3. run `coco_to_yolo.py <coco_train.json> <coco_val.json> <coco_test.json> <negative_ids.csv> <path-to-parent-folder-of-images-folder> <output-filename>`
   - coco_train.json, coco_val.json and coco_test.json are the JSON file obtained from the previous step.
   - negative_ids.csv is the optional file obtained from step 1.
   - path-to-parent-folder-of-images-folder is the parent folder of the images folder. Another folder called labels will be created with subfolders named train/val/test containing labels of the localizations. This folder structure should be maintained for the Yolo network to run.
   - output-filename user defined path with filename to store the yaml file generated.
4. Refer to YoloV5 documentation to run the yolo network.
5. run `create_strategy_yaml.py <img_size> <type_id> <version_id> <weights_path> <data_path> <output>`
   - img_size is the dimension of the image frame to be used for detection. Depends on the trained value ex.640.
   - type_id is the id of the annotation type. Can be found in tator. Differs across the projects.
   - version_id is the id of the version. Can be found in tator. Differs across the projects.
   - weights_path is the path to the weights that needs to be used for detection. It is generally the weights/best.pt generated after training yolo.
   - data_path is the path to the yaml file containing info on the dataset. This is the file created from coco_to_yolo.py.
   - output is the path including the filename to store the yaml structure.