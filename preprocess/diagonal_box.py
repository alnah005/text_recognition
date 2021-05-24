# -*- coding: utf-8 -*-
"""
file: diagonal_box.py

@author: Suhail.Alnahari

@description: 

@created: 2021-05-20T21:24:18.734Z-05:00

@last-modified: 2021-05-24T15:46:41.438Z-05:00
"""

# standard library
# 3rd party packages
# local source

import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    print("width: {}, height: {}".format(width, height))

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))

    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


def get_rotated_sample(coords, image_path=None, img=None, visualize=False, angle=0):
    assert image_path is not None or img is not None

    if img is None:
        try:
            img = Image.open(image_path)
        except:
            print("Image not found")
            return None
    rotated_img = rotate_image(np.asarray(img), angle)
    mask_bbox = np.zeros(np.asarray(img).shape)
    assert len(coords) == 8
    cv2.fillPoly(mask_bbox, [np.array([(coords['x'+str(i+1)], coords['y'+str(i+1)])
                                       for i in range(int(len(coords)/2))])], (255, 255, 255))
    rotated_bbox = rotate_image(mask_bbox, angle)

    cnts, _ = cv2.findContours(rotated_bbox.astype(
        'uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(cnts[0])
    # for axis aligned
    cx: float = rect[0][0]
    cy: float = rect[0][1]
    w: float = rect[1][0]
    h: float = rect[1][1]
    a: float = rect[2]*np.pi/180
    # box = cv2.boxPoints(rect)
    if visualize:
        plt.figure()
        plt.imshow(np.asarray(img)[min(coords['y1'], coords['y2'], coords['y3'], coords['y4']):max(coords['y1'], coords['y2'], coords['y3'], coords['y4']), min(
            coords['x1'], coords['x2'], coords['x3'], coords['x4']):max(coords['x1'], coords['x2'], coords['x3'], coords['x4'])])
        plt.show()

        plt.figure()
        plt.imshow(mask_bbox)
        plt.show()

        plt.figure()
        plt.imshow(mask_bbox[min(coords['y1'], coords['y2'], coords['y3'], coords['y4']):max(coords['y1'], coords['y2'], coords['y3']+10, coords['y4']), min(
            coords['x1'], coords['x2'], coords['x3'], coords['x4']):max(coords['x1'], coords['x2'], coords['x3'], coords['x4'])])
        plt.show()

        plt.figure()
        plt.imshow(rotated_img)
        plt.show()

        plt.figure()
        plt.imshow(rotated_bbox)
        plt.show()

        centre = np.array([cx, cy])
        original_points = np.array([[cx-0.5*w, cy-0.5*h],                # This would be the box if theta = 0
                                    [cx + 0.5*w, cy-0.5*h],
                                    [cx + 0.5*w, cy + 0.5*h],
                                    [cx - 0.5*w, cy + 0.5*h]])
        rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        corners = np.matmul(original_points - centre, rotation) + centre

        plt.figure()
        plt.imshow(cv2.drawContours(rotated_img, [
            corners.astype(int)], 0, (255, 255, 255), 10))
        plt.show()

        img_crop, _ = crop_rect(rotated_img, rect)

        # ASSUMING LINE WIDTH IS LARGER THAN ITS HEIGHT
        if img_crop.shape[0] > img_crop.shape[1]:
            plt.figure()
            plt.imshow(np.rot90(img_crop))
            plt.show()

            plt.figure()
            plt.imshow(np.rot90(np.rot90(np.rot90(img_crop))))
            plt.show()
        else:
            plt.figure()
            plt.imshow(np.rot90(np.rot90(img_crop)))
            plt.show()

            plt.figure()
            plt.imshow(img_crop)
            plt.show()
        rotated_img = rotate_image(np.asarray(img), angle)
        mask_bbox = np.zeros(np.asarray(img).shape)
        assert len(coords) == 8
        cv2.fillPoly(mask_bbox, [np.array([(coords['x'+str(i+1)], coords['y'+str(i+1)])
                                           for i in range(int(len(coords)/2))])], (255, 255, 255))
        rotated_bbox = rotate_image(mask_bbox, angle)

        cnts, _ = cv2.findContours(rotated_bbox.astype(
            'uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(cnts[0])
        # for axis aligned
        cx: float = rect[0][0]
        cy: float = rect[0][1]
        w: float = rect[1][0]
        h: float = rect[1][1]
        a: float = rect[2]*np.pi/180
    return Image.fromarray(rotated_img), [max(int(round(cx-0.5*w)), 0), max(int(round(cy-0.5*h)), 0), int(round(w)), int(round(h)), a]


if __name__ == '__main__':
    root_path = os.getcwd()+"/preprocess/"

    loc = '/'+"97add088-3c2f-44b0-9c67-17e17e28c9a3.jpeg"
    fn = "{0}{1}".format(
        root_path + "../data/ASM/Original_Images/", loc.split('/')[-1])

    line_box = (958, 26, 1538, 535)

    coords = {'x1': line_box[0], 'y1': line_box[3], 'x2': line_box[0], 'y2': line_box[1],
              'x3': line_box[2], 'y3': line_box[1], 'x4': line_box[2], 'y4': line_box[3]}

    print(get_rotated_sample(coords, image_path=fn, visualize=True, angle=0))
