import argparse
import datetime
import os
import tator
import pickle
import yaml
from pprint import pprint

def get_train_test_split(pos_locs,neg_locs,test_size = 0.4):
    '''Splits the localization into train, val and test sets, combining positive and negative samples. 
    Note that the test and val will have same split size.
    pos_locs is positive localizations and neg_locs is negative localizations'''
    length_pos_locs = len(pos_locs)
    length_neg_locs = len(neg_locs)
    train_pos_range = int(length_pos_locs - (length_pos_locs*test_size))
    val_pos_range = train_pos_range+int((length_pos_locs - train_pos_range)/2)
    
    if length_neg_locs > 0:
        train_neg_range = int(length_neg_locs - (length_neg_locs*test_size))
        val_neg_range = train_neg_range+int((length_neg_locs - train_neg_range)/2)
        
        train = pos_locs[0:train_pos_range]+neg_locs[0:train_neg_range]
        val = pos_locs[train_pos_range:val_pos_range]+neg_locs[train_neg_range:val_neg_range]
        test = pos_locs[val_pos_range:]+neg_locs[val_neg_range:]
    else:
        train = pos_locs[0:train_pos_range]
        val = pos_locs[train_pos_range:val_pos_range]
        test = pos_locs[val_pos_range:]
    
    return train,val,test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('apiToken')
    parser.add_argument('projectId')
    parser.add_argument('sectionId')
    parser.add_argument('testSize',type = float,default = 0.4,help='The propotion of train and test (val and test) set split.')
    parser.add_argument('positives',type = int,help = 'The value of CORRECTED set in tator for positive samples.')
    parser.add_argument('positive_size',type = int,default = 1000,help = 'The number of positive localizations to be saved' )
    parser.add_argument('--negatives',type = int,help = 'The value of CORRECTED set in tator for negative samples.')
    parser.add_argument('--negative_set_size',type = float,default = 0.1,help='The proportion negative to positive samples.')
    parser.add_argument('--folder_path',type = str,default='',help='An folder path to save the CSV and PKL files and to create the folder structure needed for yolov5')
    args = parser.parse_args()
    
    token = args.apiToken
    projectId = args.projectId
    sectionId = args.sectionId
    positive_size = args.positive_size
    positive_corrections = str(args.positives)
    negative_set_size = args.negative_set_size
    if args.negatives:
        negative_corrections = str(args.negatives)
    else:
        negative_corrections = None
    testSize = args.testSize
    
    if args.folder_path:
        path = args.folder_path
    else:
        path = os.getcwd()
    
    if os.path.exists(path) == False:
        os.makedirs(path)
        
    ROOT = 'Data_iter'
    n=1
    base = ROOT + str(n)

    while n>0:
        sub_dir = os.path.join(path,base)
        if os.path.exists(sub_dir) == False:
            os.mkdir(sub_dir)
            for i in ['images','labels']:
                sub_dir2 = os.path.join(sub_dir,i)
                sub_train = os.path.join(sub_dir2,'train')
                sub_val = os.path.join(sub_dir2,'val')
                sub_test = os.path.join(sub_dir2,'test')
                os.mkdir(sub_dir2)
                os.mkdir(sub_train)
                os.mkdir(sub_val)
                os.mkdir(sub_test)
            break
        else:
            n+=1
            base = ROOT + str(n)
    
    api = tator.get_api(token = token)
    
    media_list = api.get_media_list(projectId,section=sectionId)
    media_ids = [x.id for x in media_list]
    
    pos_localisations = api.get_localization_list(projectId, 
                                        section = sectionId, 
                                        media_id = media_ids,  
                                        attribute = [f'Corrected::{str(positive_corrections)}'])
    pos_localisations = pos_localisations[0:positive_size]
    if negative_corrections:
        neg_localisations = api.get_localization_list(projectId, 
                                        section = sectionId, 
                                        media_id = media_ids,  
                                        attribute = [f'Corrected::{str(negative_corrections)}'])
        
        neg_set = min(int(len(pos_localisations)*negative_set_size),len(neg_localisations))
        # print(neg_set)
        neg_localisations = neg_localisations[0:neg_set]
        neg_ids = [str(x.id)+'_'+str(x.media) for x in neg_localisations]
        str_ids = ','.join(neg_ids)
        with open(f'{path}/negative_ids.csv','w') as f:
            f.write(str_ids)
    else:
        neg_localisations = []
        
    train,val,test = get_train_test_split(pos_localisations,neg_localisations,test_size=testSize)
    
    with open(f'{path}/train_locs.pkl','wb') as f:
        pickle.dump(train,f)
        
    with open(f'{path}/val_locs.pkl','wb') as f:
        pickle.dump(val,f)
        
    with open(f'{path}/test_locs.pkl','wb') as f:
        pickle.dump(test,f)
        
    args_yaml = {'apiToken':token,
                 'train':f'{path}/train_locs.pkl',
                 'val':f'{path}/val_locs.pkl',
                 'test':f'{path}/test_locs.pkl',
                 'cocoPath':path,
                 'imagesPath':f'{sub_dir}/images'
    }
    with open(f'{path}/locsImgCoco.yaml', 'w') as fp:
        yaml.dump(args_yaml, fp)
    

if __name__=="__main__":
    main() 
    