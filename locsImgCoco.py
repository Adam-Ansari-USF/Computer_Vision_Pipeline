import argparse
import datetime
import os
import tator
import pickle
import json
import shutil
import cv2
import yaml
from pprint import pprint


def get_images(api,localizations, path, condition='train'):
    """
    This function downloads the images corresponding to the input localizations from the server and saves them on disk.

    Args:
        api (Tator api): tator.get_api(token)
        localizations (list): A list of localizations
        path (str): The path to the directory where the images will be saved
        condition (str, optional): The type of the data set. Defaults to 'train'.

    Returns:
        None
    """
    ft = os.path.join(path, condition) # Create a path for the images using the specified directory and dataset condition
    for x in localizations:
        media = x 
        name = media.id # Get the ID of the media
        date_captured = str(media.modified_datetime) # Get the modification datetime of the media as a string
        height = media.height # Get the height of the media
        width = media.width # Get the width of the media
        image_id = media.media # Get the ID of the image
        filename = f"{name}_{image_id}.png" # Create a filename for the image by combining the media ID and the image ID
        if os.path.exists(ft) == False: # Check if the directory to save the images exists, and create it if not
            os.makedirs(ft)
        on_disk_filename = os.path.join(ft, filename) # Create a path to save the image on disk by joining the path and filename
        temp_path = api.get_frame(x.media, frames=[x.frame]) # Download the image from the server using the Tator API
        shutil.move(temp_path, on_disk_filename) # Move the downloaded image to the specified path and filename

def convert_to_coco(localizations, imagePath, cocoPath, condition='train'):
    """
    This function converts a list of localizations into COCO annotation format and saves it to a JSON file.

    Args:
        localizations (list): A list of localizations
        imagePath (str): The path to the directory containing the images
        cocoPath (str): The path to the directory to save the COCO JSON file
        condition (str, optional): The type of the data set. Defaults to 'train'.

    Returns:
        None
    """
    dict_info = {}
    
    ### licenses section
    list_licenses = []
    license = dict()
    license['url'] = 'https://www.tatorapp.com/licenses'
    license['id'] = 1
    license['name'] = 'Attribution-Commercial-ShareLicense'
    list_licenses.append(license)

    list_images = []
    list_categories = []
    list_annotations = []

    ### categories section  ## modified
    category = {}
    category['supercategory'] = 'vessel'
    category['id'] = 1
    category['name'] = 'small fishery boat'
    list_categories.append(category)

    image_seen = {} # modified
    image_path = os.path.join(imagePath,condition)
    
    for i,local in enumerate(localizations):
        image_id = local.id
        frame = local.frame
        media_id = local.media
        filename_creation = image_path + "/" + str(image_id) + "_" + str(media_id) + ".png"

        img = cv2.imread(filename_creation, cv2.IMREAD_UNCHANGED)
        coco_width = img.shape[1]
        coco_height = img.shape[0]

        ### info section

        dict_info['description'] = 'Tator 2021 Dataset'
        dict_info['url'] = 'https://www.tatorapp.com/'
        dict_info['version'] = '1.0'
        dict_info['contributor'] = 'EDF, CVision'
        dict_info['date_created'] = str(local.created_datetime.date()).replace('-','/')

        ### images section

        image = dict()
        url = "https://www.tatorapp.com/rest/GetFrame/"+ str(local.media) + "?" + "frames=" + str(local.frame)
        image['license'] = 1
        file_name = str(local.media) + '_' + str(local.frame) + ".png" ## modified
        image['file_name'] = file_name ### modified
        if file_name not in image_seen: ## modified
            image['coco_url'] = url
            image['height'] = coco_height
            image['width'] = coco_width
            image['date_captured'] = str(local.modified_datetime)[:19]
            image['flicker_url'] = url
            image['id'] = local.id
            image_seen[file_name] = local.id ### modified
            list_images.append(image)

        ### annotation section
        annotation = {}
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_seen[file_name]  ## modified

        ## Transforming custom bbox to CoCo bbox format
        coco_width_bbox = local.width * coco_width ## converting the width ratio to absolute values
        coco_height_bbox = local.height * coco_height ## converting the height ration to absolute values
        xleft = (local.x * coco_width) ## getting left coordinate of x 
        yleft = (local.y * coco_height) ## getting left coordinate of y
        bbox = [xleft, yleft, coco_width_bbox, coco_height_bbox] ## creating bbox list
        annotation['bbox'] = bbox
        annotation['category_id'] = 1
        annotation['id'] = i+1
        #print(annotation)
        list_annotations.append(annotation)
        
    ##Creating COCO structure and loading them into a JSON
    coco_structure = {
    "info": dict_info,
    "licenses": list_licenses,
    "images":list_images,
    "categories":list_categories,
    "annotations":list_annotations
    }
    
    fileName =  cocoPath + f'/fog_it2_{condition}.json'

    with open(fileName, "w") as f:
        json.dump(coco_structure, f)
    
    
         
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('argsYaml',help='The path to the yaml containing arguments created in the previous step')
    args = parser.parse_args()
    
    with open(args.argsYaml,'rb') as f:
        args_yaml = yaml.safe_load(f)
        
    train_path = args_yaml['train']
    val_path = args_yaml['val']
    test_path = args_yaml['test']
    
    with open(train_path,'rb') as f:
        train = pickle.load(f)
        
    with open(test_path,'rb') as f:
        test = pickle.load(f)
        
    with open(val_path,'rb') as f:
        val = pickle.load(f)
    
    token = args_yaml['apiToken']
    api = tator.get_api(token = token)
    
    imagePath = args_yaml['imagesPath']
    cocoPath = args_yaml['cocoPath']
        
    get_images(api,train,imagePath,'train')
    get_images(api,test,imagePath,'test')
    get_images(api,val,imagePath,'val')
    
    convert_to_coco(train,imagePath,cocoPath,'train')
    convert_to_coco(test,imagePath,cocoPath,'test')
    convert_to_coco(val,imagePath,cocoPath,'val')
    
if __name__=="__main__":
    main()
    