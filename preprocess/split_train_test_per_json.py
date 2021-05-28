# -*- coding: utf-8 -*-
"""
file: split_train_test_per_json.py

@author: Suhail.Alnahari

@description: 

@created: 2021-05-28T10:00:54.493Z-05:00

@last-modified: 2021-05-28T10:16:13.328Z-05:00
"""

# standard library
import os
from typing import Dict, List, Any
# 3rd party packages
import json
import numpy as np
import shutil
# local source


ImageDir = "Original_Images_rotated/"
dataset = "/ASM/"
labels_val = "final_annotations_val.json"
labels_train = "final_annotations_train.json"
validation_percentage = 0.2
try:
    root = os.getcwd()
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.isfile(root+"/data"+dataset+labels_val)
    assert os.path.isfile(root+"/data"+dataset+labels_train)
except:
    root = "../"
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.isfile(root+"/data"+dataset+labels_val)
    assert os.path.isfile(root+"/data"+dataset+labels_train)

try:
    shutil.rmtree(root+"/data"+dataset+ImageDir[:-1]+"_val")
    print("overwriting existing output dir")
except:
    print("first time creating dir")

try:
    shutil.rmtree(root+"/data"+dataset+ImageDir[:-1]+"_train")
    print("overwriting existing output dir")
except:
    print("first time creating dir")


os.mkdir(root+"/data"+dataset+ImageDir[:-1]+"_val")
assert os.path.exists(root+"/data"+dataset+ImageDir[:-1]+"_val")

os.mkdir(root+"/data"+dataset+ImageDir[:-1]+"_train")
assert os.path.exists(root+"/data"+dataset+ImageDir[:-1]+"_train")

with open(root+"/data"+dataset+labels_val) as json_file:
    angular_labels_dict = json.load(json_file)

print(f"moving {len(angular_labels_dict['images'])} images to validation dir")

for i in angular_labels_dict['images']:
    try:
        shutil.copy(root+"/data"+dataset+ImageDir +
                    i['file_name'], root+"/data"+dataset+ImageDir[:-1]+"_val/"+i['file_name'])
    except:
        print("error", i['file_name'])

with open(root+"/data"+dataset+labels_train) as json_file:
    angular_labels_dict = json.load(json_file)

print(f"moving {len(angular_labels_dict['images'])} images to train dir")

for i in angular_labels_dict['images']:
    try:
        shutil.copy(root+"/data"+dataset+ImageDir +
                    i['file_name'], root+"/data"+dataset+ImageDir[:-1]+"_train/"+i['file_name'])
    except:
        print("error", i['file_name'])
