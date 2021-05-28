# -*- coding: utf-8 -*-
"""
file: add_image_height_width_to_json.py

@author: Suhail.Alnahari

@description: 

@created: 2021-05-28T15:47:41.812Z-05:00

@last-modified: 2021-05-28T15:58:18.041Z-05:00
"""

# standard library
import os
from typing import Dict, List, Any
# 3rd party packages
import json
import numpy as np
import shutil
import cv2
# local source


ImageDir = "Original_Images_rotated/"
dataset = "/ASM/"
labels_val = "final_annotations_val.json"
labels_train = "final_annotations_train.json"
try:
    root = os.getcwd()
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.exists(root+"/data"+dataset+ImageDir)
    assert os.path.isfile(root+"/data"+dataset+labels_val)
    assert os.path.isfile(root+"/data"+dataset+labels_train)
except:
    root = "../"
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.exists(root+"/data"+dataset+ImageDir)
    assert os.path.isfile(root+"/data"+dataset+labels_val)
    assert os.path.isfile(root+"/data"+dataset+labels_train)

with open(root+"/data"+dataset+labels_val) as json_file:
    final_labels = json.load(json_file)

for i in range(len(final_labels["images"])):
    im = cv2.imread(root+"/data"+dataset+ImageDir +
                    final_labels["images"][i]["file_name"])
    assert len(im.shape) > 1
    if len(im.shape) == 3:
        height, width, _ = im.shape
    elif len(im.shape) == 2:
        height, width = im.shape
    else:
        print("error at", root+"/data"+dataset+ImageDir +
              final_labels["images"][i]["file_name"])
        continue
    final_labels["images"][i]["height"] = int(height)
    final_labels["images"][i]["width"] = int(width)

with open(root+"/data"+dataset+labels_val, 'w') as fp:
    json.dump(final_labels, fp, indent=4)

with open(root+"/data"+dataset+labels_train) as json_file:
    final_labels = json.load(json_file)

for i in range(len(final_labels["images"])):
    im = cv2.imread(root+"/data"+dataset+ImageDir +
                    final_labels["images"][i]["file_name"])
    assert len(im.shape) > 1
    if len(im.shape) == 3:
        height, width, _ = im.shape
    elif len(im.shape) == 2:
        height, width = im.shape
    else:
        print("error at", root+"/data"+dataset+ImageDir +
              final_labels["images"][i]["file_name"])
        continue
    final_labels["images"][i]["height"] = int(height)
    final_labels["images"][i]["width"] = int(width)

with open(root+"/data"+dataset+labels_train, 'w') as fp:
    json.dump(final_labels, fp, indent=4)
