# -*- coding: utf-8 -*-
"""
file: convert_angular_labels_to_coco.py

@author: Suhail.Alnahari

@description: 

@created: 2021-05-25T09:27:28.391Z-05:00

@last-modified: 2021-05-25T10:27:23.182Z-05:00
"""

# standard library
import os
from typing import Dict, List, Any
# 3rd party packages
import json
import numpy as np
# local source


ImageDir = "Original_Images_rotated/"
dataset = "/ASM/"
labels = "angular_labels.csv"
validation_percentage = 0.2
try:
    root = os.getcwd()
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.isfile(root+"/data"+dataset+labels)
except:
    root = "../"
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.isfile(root+"/data"+dataset+labels)

f = open(root+"/data"+dataset+"angular_labels.csv", 'r')
final_labels_train: Dict[str, List[Dict[str, Any]]] = {
    "images": [], "annotations": [], "categories": []}
final_labels_val: Dict[str, List[Dict[str, Any]]] = {
    "images": [], "annotations": [], "categories": []}
images_dict: Dict[str, Dict[str, Any]] = {}
annotations_id = 0
image_id = 0
for row in f:
    try:
        useful_cols = row.replace('\n', '').split(
            ',')[:6]
        assert os.path.isfile(root+"/data"+dataset+ImageDir+useful_cols[0])
        if images_dict.get(useful_cols[0]) is None:
            images_dict[useful_cols[0]] = {
                'id': image_id, "isVal": np.random.uniform() < validation_percentage}
            image_id += 1
        xmin = float(useful_cols[1])
        ymin = float(useful_cols[2])
        w = float(useful_cols[3])
        h = float(useful_cols[4])
        theta = float(useful_cols[5])
        centre = np.array([xmin + w / 2.0, ymin + h / 2.0])
        original_points = np.array([[xmin, ymin],                # This would be the box if theta = 0
                                    [xmin + w, ymin],
                                    [xmin + w, ymin + h],
                                    [xmin, ymin + h]])
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        corners = np.matmul(original_points - centre, rotation) + centre
        annot = {
            "id": annotations_id,
            "image_id": images_dict[useful_cols[0]]['id'],
            "category_id": 1,
            # all floats, where theta is measured in radians anti-clockwise from the x-axis.
            "bbox": [xmin, ymin, w, h, theta],
            # Required for validation scores.
            "segmentation": [[corners[0][0], corners[0][1], corners[1][0], corners[1][1], corners[2][0], corners[2][1], corners[3][0], corners[3][1]]],
            "area": w*h,  # w * h. Required for validation scores
            "iscrowd": 0  # Required for validation scores
        }
        if images_dict[useful_cols[0]]['isVal']:
            final_labels_val["annotations"].append(annot)
        else:
            final_labels_train["annotations"].append(annot)
        annotations_id += 1
    except:
        print("Error encountered", row)

for i in images_dict:
    image_dict = {
        "id": images_dict[i]['id'],
        "file_name": i
    }
    if images_dict[i]['isVal']:
        final_labels_val["images"].append(image_dict)
    else:
        final_labels_train["images"].append(image_dict)

categs = {
    "id": 1,
    "name": "textline"
}

final_labels_val["categories"].append(categs)
final_labels_train["categories"].append(categs)

with open(root+"/data"+dataset+'final_annotations_train.json', 'w') as fp:
    json.dump(final_labels_train, fp, indent=4)

with open(root+"/data"+dataset+'final_annotations_val.json', 'w') as fp:
    json.dump(final_labels_val, fp, indent=4)

num_images_train, num_annotation_train = len(
    final_labels_train["images"]), len(final_labels_train["annotations"])
num_images_val, num_annotation_val = len(
    final_labels_val["images"]), len(final_labels_val["annotations"])
print(
    f"Num of Images,Annotations for train: {num_images_train},{num_annotation_train}\nNum of Images,Annotations for val: {num_images_val},{num_annotation_val}")
