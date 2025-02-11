import numpy as np
import pandas as pd
import os
import sys
import pickle

from PIL import Image
import PIL
import requests
from io import BytesIO
import time

train_lines = []
img_files = []
count = 0

def readImg(url, grey=True):
    global count
    if (count+1) % 5000 == 0:
        time.sleep(600)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert("L")
    return img

# create a batch with the next group of data from ASM
def create_ASM_batch(resize_to=0.5):
    global train_lines
    global img_files
    global count
    print(count)
    # Read in all classifications
    sv_fold = os.path.sep.join(__file__.split(os.path.sep)[:-1])
    if len(sv_fold) > 0:
       sv_fold += "/../data/ASM/"
    else:
       sv_fold = "./../data/ASM/"
    if not os.path.isdir(sv_fold+"Images"):
        os.mkdir(sv_fold+"Images")

    root_path = os.path.sep.join(__file__.split(os.path.sep)[:-1])
    if len(root_path) > 0:
        root_path += '/'
    else:
        root_path = './'

    print("Loading classification data", flush=True)
    if os.path.exists(root_path+"../data/ASM/full_train.csv"):
        data = pd.read_csv(root_path+"../data/ASM/full_train.csv", sep="\t")
    else:
        print("File doesn't exist, run \"preprocess_ASM_csv.py\" first", flush=True)

    # Preprocess by splitting image into sections
    # load input_shape from file output by preprocess
    with open(root_path+"../data/img_size.txt", "r") as f:
        doc = f.readline()
        w, h = doc.split(",")
        maxh = int(float(h))
        maxw = int(float(w))

    onepercent = len(data)//100
    tenpercent = onepercent*10

    print("Creating image files and training csv", flush=True)
    for i in range(count, len(data)):
        curclas = data.iloc[i].copy()
        loc = curclas["location"]
        fn = "{0}{1}".format(root_path+ "../data/ASM/Original_Images/",loc.split('/')[-1])
        try:
            img = Image.open(fn)
        except:
            img = readImg(loc)
            img.save(fn)
        trans = curclas["transcription"]
        # extract area in bounding box
        line_box = eval(curclas["line_box"])
        img_line = img.crop(line_box)

        
        #img_line = img_line.resize([int(j) for j in np.floor(np.multiply(resize_to, img_line.size))])
        img_line_np = np.array(img_line)

        # turn everything above previous line white - slope top
        m = curclas["slope_top"]
        b = curclas["intercept_top"]
        for x in range(img_line_np.shape[1]):
            img_line_np[:int(np.floor(m*x+b)),x] = 255

        # turn everything below transcription line white - slope bottom
        m = curclas["slope_bottom"]
        b = curclas["intercept_bottom"]
        for x in range(img_line_np.shape[1]):
            img_line_np[int(np.ceil(m*x+b)):,x] = 255

        img_line = Image.fromarray(img_line_np)

        # resize if it's too big
        if img_line.size[1] > maxh:
            ratio = maxh / float(img_line.size[1])
            wnew = int(float(img_line.size[0]) * float(ratio))
            img_line = img_line.resize((wnew, maxh), PIL.Image.ANTIALIAS)
        if img_line.size[0] > maxw:
            ratio = maxw / float(img_line.size[0])
            hnew = int(float(img_line.size[1]) * float(ratio))
            img_line = img_line.resize((maxw, hnew), PIL.Image.ANTIALIAS)
        fn = "{0}{1}_{2}_{3}_{4}.png".format(root_path+ "../data/ASM/Images/", curclas["subject_id"], curclas["classification_id"],
                                             curclas["frame"], curclas["j"])
        # save image
        img_line.save(fn)
        img_files.append(fn)
        # add line for training
        train_lines.append(trans)

        count += 1
        if count % onepercent == 0:
            if count % tenpercent == 0:
                perc = count//onepercent
                print(str(perc)+"%", end="", flush=True)
            else:
                print(".", end="", flush=True)

    # end of loop
    savedata = pd.DataFrame.from_dict({"new_img_path":img_files, "transcription":train_lines})
    savedata = savedata[np.logical_not(savedata.duplicated())]
    savedata.to_csv(root_path+ "../data/ASM/train.csv", sep="\t", index=False)
    print("\nTraining file and {0} images created".format(len(savedata)), flush=True)
    return


if __name__ == "__main__":
    resize_to=1.0
    if len(sys.argv) >= 2:
        resize_to=float(sys.argv[1])

    redo = True
    while redo:
        try:
            create_ASM_batch(resize_to=resize_to)
            redo = False
        except:
            count += 1
            time.sleep(60)
            print("Error during batch creation, redoing", flush=True)
            redo = True
            