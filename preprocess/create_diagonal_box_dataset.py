# -*- coding: utf-8 -*-
"""
file: create_diagonal_box_dataset.py

@author: Suhail.Alnahari

@description:

@created: 2021-05-24T11:29:35.906Z-05:00

@last-modified: 2021-05-24T16:01:51.235Z-05:00
"""

# standard library
import os
from ast import literal_eval as make_tuple
from typing import Dict
# 3rd party packages
import pandas as pd
# local source
try:
    from preprocess.diagonal_box import get_rotated_sample
except:
    import diagonal_box
    get_rotated_sample = diagonal_box.get_rotated_sample

dataset = "/ASM/"
labels = "full_train.csv"
chunksize = 10 ** 6
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

f = open(root+"/data"+dataset+"angular_labels.csv", 'w')
# errors = open(root+"/data"+dataset+"label_errors.csv", 'w')
# errors.write(','.join(['img_path', 'x1', 'x2', 'x2', 'y2', 'x3', 'y3', 'x4',
#                        'y4', 'angle_deg', 'x1_orig', 'y1_orig', 'x2_orig', 'y2_orig'])+'\n')
f.write(','.join(['img_name', 'x', 'y', 'w', 'h', 'a_radian',
                  'x1_orig', 'y1_orig', 'x2_orig', 'y2_orig', 'angle_transform'])+'\n')
multiplier = 1
angles: Dict[str, int] = {}
hits = 0
reruns = 0
misses = 0
for chunk in pd.read_csv(root+"/data"+dataset+labels,
                         sep="\t", chunksize=chunksize):
    chunk['original_image_path'] = chunk['location'].apply(lambda loc: "{0}{1}".format(root+"/data"+dataset +
                                                                                       "Original_Images/", loc.split('/')[-1]))
    chunk['image_id_path'] = chunk['location'].apply(
        lambda loc: loc.split('/')[-1])
    chunk['line_box_tuples'] = chunk['line_box'].apply(
        lambda box: make_tuple(box))

    for index, row in chunk.iterrows():
        fn = row['original_image_path']
        if fn in angles.keys():
            angle_rotate = angles[fn]
        else:
            angle_rotate = (index % 90)*multiplier
            multiplier *= -1
            angles[fn] = angle_rotate

        line_box = row['line_box_tuples']
        coords = {'x1': line_box[0], 'y1': line_box[3], 'x2': line_box[0], 'y2': line_box[1],
                  'x3': line_box[2], 'y3': line_box[1], 'x4': line_box[2], 'y4': line_box[3]}
        try:
            tries = 0
            rotated_labels = [-1, -1]
            while (rotated_labels[0] <= 0 or rotated_labels[1] <= 0 or rotated_labels[2] < 0 or rotated_labels[3] < 0) and tries < 3:
                img, rotated_labels = get_rotated_sample(
                    coords, image_path=fn, angle=angle_rotate)
                tries += 1
            assert rotated_labels[0] >= 0 and rotated_labels[1] >= 0 and rotated_labels[2] > 0 and rotated_labels[3] > 0
            if not os.path.exists(root+"/data"+dataset + "Original_Images_rotated/"+row['image_id_path']):
                img.save(root+"/data"+dataset +
                         "Original_Images_rotated/"+row['image_id_path'])
            f.write(
                ','.join([row['image_id_path']]+[str(j) for j in rotated_labels]+[str(k) for k in line_box]+[str(angle_rotate)])+'\n')
            hits += 1
        except:
            misses += 1
            co = [coords['x1'], coords['y1'], coords['x2'], coords['y2'], coords['x3'],
                  coords['y3'], coords['x4'], coords['y4']]
            # errors.write(','.join(
            #     [fn]+[str(c) for c in co]+[str(angle_rotate)]+[str(k) for k in line_box])+'\n')
            print(','.join(
                [fn]+[str(c) for c in co]+[str(angle_rotate)]+[str(k) for k in line_box]))
f.close()
# errors.close()


print(
    f"number of converted with no problems = {hits}\nnumber of converted with some problems {reruns}\nnumber of errors {misses}")
