# -*- coding: utf-8 -*-
"""
file: split_asm_train_test.py

@author: Suhail.Alnahari

@description: 

@created: 2021-05-27T11:42:08.441Z-05:00

@last-modified: 2021-05-27T12:44:37.382Z-05:00
"""

# standard library
import os
from typing import Dict, List, Any
# 3rd party packages
import json
import numpy as np
import pandas as pd
import shutil
# local source


angular_labels_validation_file = "final_annotations_val.json"

train_file = "train.csv"
dataset = "/ASM/"
labels = "full_train.csv"
outputDataset = "/ASMSplit/"
images_dir = "Images/"
alternative = "Images_2/"
path_prefix = "/home/fortson/alnah005/text_recognition/data/ASM/"
try:
    root = os.getcwd()
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.exists(root+"/data"+dataset+images_dir)
    assert os.path.isfile(root+"/data"+dataset+labels)
    assert os.path.isfile(root+"/data"+dataset+train_file)
    assert os.path.isfile(root+"/data"+dataset+angular_labels_validation_file)
except:
    root = "../"
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.exists(root+"/data"+dataset+images_dir)
    assert os.path.isfile(root+"/data"+dataset+labels)
    assert os.path.isfile(root+"/data"+dataset+train_file)
    assert os.path.isfile(root+"/data"+dataset+angular_labels_validation_file)
try:
    shutil.rmtree(root+"/data"+outputDataset)
    print("overwriting existing output dir")
except:
    print("first time creating dir")

os.mkdir(root+"/data"+outputDataset)
assert os.path.exists(root+"/data"+outputDataset)

train_pd = pd.read_csv(root+"/data"+dataset+train_file, delimiter='\t')
with open(root+"/data"+dataset+angular_labels_validation_file) as json_file:
    angular_labels_dict = json.load(json_file)
full_train_pd = pd.read_csv(root+"/data"+dataset+labels, delimiter='\t')

full_train_pd['image'] = full_train_pd['location'].apply(
    lambda loc: "{0}".format(loc.split('/')[-1]))
val_images = [i['file_name'] for i in angular_labels_dict['images']]

train_pd['file_name'] = train_pd['new_img_path'].apply(
    lambda name: "{0}".format(name.split('\\')[-1].split('/')[-1]))
train_pd['key'] = train_pd['file_name'].apply(
    lambda name: "{0}".format('_'.join(name.split('_')[:3])))

train_pd['new_img_path'] = train_pd['file_name'].apply(
    lambda name: path_prefix+images_dir+name if os.path.isfile(path_prefix+images_dir+name) else path_prefix+alternative+name)

full_train_pd['key'] = full_train_pd.apply(
    lambda row: "{0}".format(str(row.subject_id)+'_'+str(row.classification_id)+'_'+str(row.frame)), axis=1)
full_train_pd_val = full_train_pd.loc[full_train_pd['image'].isin(val_images)]
full_train_pd_train = full_train_pd.loc[~full_train_pd['image'].isin(
    val_images)]

train_pd_val = train_pd.loc[train_pd['key'].isin(
    full_train_pd_val['key'].tolist())][['new_img_path', 'transcription']]
train_pd_train = train_pd.loc[train_pd['key'].isin(
    full_train_pd_train['key'].tolist())][['new_img_path', 'transcription']]

train_pd_val.to_csv(root+"/data"+outputDataset +
                    "val.csv", sep="\t", index=False)
train_pd_train.to_csv(root+"/data"+outputDataset +
                      "train.csv", sep="\t", index=False)
