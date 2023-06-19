import argparse
import json
import yaml
import os
import cv2
import tqdm

from collections import defaultdict

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train',help='The path to the json file containing images in COCO format of the train set')
    parser.add_argument('val',help='The path to the json file containing images in COCO format of the val set')
    parser.add_argument('test',help='The path to the json file containing images in COCO format of the test set')
    parser.add_argument('negative_frameIds',help='The path to the CSV file containing the ids of the negative frame that needs to be excluded while creating labels.')
    parser.add_argument('path',help= 'The path to the parent folder containing the images and labels folder')
    parser.add_argument('output',help='The path including the filename to be created to store the coco structure as a yaml.')
    args = parser.parse_args()
    
    with open(args.train, 'r') as fp:
        coco_object_train = json.load(fp)
    
    with open(args.val, 'r') as fp:
        coco_object_val = json.load(fp)
        
    with open(args.test, 'r') as fp:
        coco_object_test = json.load(fp)
        
    with open(args.negative_frameIds,"r") as f:
            frame_ids = f.read().split(',')
            frame_ids = set(frame_ids)
    
    
    yolov5_obj={'path': args.path,
                'train': 'images/train', #TODO split
                'val': 'images/val', #TODO split
                'test': 'images/test',
                'nc': 0,
                'names': []}
    
    for i in ['train','val','test']:
        if i == 'train':
            coco_object = coco_object_train
        elif i == 'val':
            coco_object = coco_object_val
        else:
            coco_object = coco_object_test
        coco_cat_to_yolo={}
        for category in coco_object['categories']:
            coco_cat_to_yolo[category['id']] = yolov5_obj['nc']
            if category['name'] not in yolov5_obj['names']:
                yolov5_obj['nc'] += 1
                yolov5_obj['names'].append(category['name'])

        annotations_by_image = defaultdict(lambda: [])
        for annotation in coco_object['annotations']:
            annotations_by_image[annotation['image_id']].append(annotation)

        # data_dir = args.conditions + '_data' 
        
        for image in tqdm.tqdm(coco_object['images'], desc='images'):
            image_name = str(image['id'])+'_'+os.path.splitext(image['file_name'])[0].split('_')[0]
            if image_name not in frame_ids:
                # image_name = str(image['id'])+'_'+os.path.splitext(image['file_name'])[0].split('_')[0]
                label_path=os.path.join(f"{args.path}/labels/{i}/{image_name}")
                with open(f"{label_path}.txt",'w') as fp:
                    for annotation in annotations_by_image[image['id']]:
                            if 'width' in image and 'height' in image:
                                image_width = image['width']
                                image_height = image['height']
                            else:
                                img=cv2.imread(os.path.join('images',image['file_name']))
                                shape = img.shape
                                image_width = shape[1]
                                image_height = shape[0]

                           # Yolov5 is relative Center X Center Y width height
                            print(f"Image dims = {image_width}x{image_height}")
                            print(f"bbox = {annotation['bbox']}")
                            rel_x = annotation['bbox'][0] / image_width
                            rel_y = annotation['bbox'][1] / image_height
                            rel_width = annotation['bbox'][2] / image_width
                            rel_height = annotation['bbox'][3] / image_height
                            center_x = rel_x + ((rel_width) / 2)
                            center_y = rel_y + ((rel_height) / 2)
                            print(f"x={rel_x}, y={rel_y}, width={rel_width}, heighht={rel_height}, center_x={center_x}, center_y={center_y}")
                            fp.write(f"{coco_cat_to_yolo[annotation['category_id']]} {center_x} {center_y} {rel_width} {rel_height}\n")
                    
    

    with open(args.output, 'w') as fp:
        yaml.dump(yolov5_obj, fp)