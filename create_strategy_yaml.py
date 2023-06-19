import argparse
import json
import yaml
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_size',help='The dimension of the image frame to be used for detection. Depends on the trained value ex.640')
    parser.add_argument('type_id',help='The id of the annotation type. Can be found in tator. Differs across the projects.')
    parser.add_argument('version_id',help='The id of the version. Can be found in tator. Differs across the projects.')
    parser.add_argument('weights_path',help='The path to the weights that needs to be used for detection. It is generally the weights/best.pt generated after training yolo.')
    parser.add_argument('data_path',help='The path to the yaml file containing info on the dataset. This is the file created from coco_to_yolo.py')
    parser.add_argument('output',help='The path including the filename to store the yaml structure.')
    args = parser.parse_args()


    data_yml = {
            "image_size": int(args.img_size),
            "localization_type_id": int(args.type_id),
            "weights": args.weights_path,
            "data": args.data_path,
            "label_attribute": "Label",
            "conf_attribute": "Confidence",
            "version_id": int(args.version_id)
     }
        
    with open(args.output, 'w') as fp:
        yaml.dump(data_yml,fp)