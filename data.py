import os 
import glob
from PIL import Image
import numpy as np
from IPython.display import display


cats = glob.glob('./train_data/train_data/*')
#cats = [os.path.basename(cat) for cat in cats]

path = 'MechanicalToolsClassification/data/{:05d}.jpg'
counter = 1
datain_tr = []
dataou_tr = []
datain_vl = []
dataou_vl = []

# num = 0
# for i0, cat in enumerate(cats):
#     name = os.path.basename(cat)
#     images = images = list(set(sorted(glob.glob(os.path.join(cat, '*.jpg'))) +   sorted(glob.glob(os.path.join(cat, '*.JPEG'))) +   sorted(glob.glob(os.path.join(cat, '*.png')))))
#     num = num + len(images)
#     print(cat, len(images))


# ind = np.random.permutation(num)
# sdfsf=sfsfsfdsf

for i0, cat in enumerate(cats):
    name = os.path.basename(cat)

    images = np.array(list(set(sorted(glob.glob(os.path.join(cat, '*.jpg'))) +   sorted(glob.glob(os.path.join(cat, '*.JPEG'))) +   sorted(glob.glob(os.path.join(cat, '*.png'))))))
    num = len(images)
    ind = np.random.permutation(num)
    images = images[ind]
    images_vl = images[:40]
    images_tr = images[40:] 

    for image in images_vl:
        img = Image.open(image)
        img_new = img.resize((64,64), Image.Resampling.NEAREST)
        if len(np.array(img_new).shape) == 3 and np.array(img_new).shape[-1] == 3:
            datain_vl.append(np.expand_dims(np.array(img_new), axis = 0))
            dataou_vl.append(i0)
            print("{:05d} | {:05d} - CAT: {} ".format(counter, len(images), name))
            counter = counter + 1

        
    for image in images_tr:
        img = Image.open(image)
        img_new = img.resize((64,64), Image.Resampling.NEAREST)
        if len(np.array(img_new).shape) == 3 and np.array(img_new).shape[-1] == 3:
            datain_tr.append(np.expand_dims(np.array(img_new), axis = 0))
            dataou_tr.append(i0)
            print("{:05d} | {:05d} - CAT: {} ".format(counter, len(images), name))
            counter = counter + 1
    
   
print(len(datain_tr))
for d in datain_tr:
    if d.shape != (1, 64, 64, 3):
        print(d.shape)

datain_tr = np.concatenate(datain_tr, axis = 0)
datain_vl = np.concatenate(datain_vl, axis = 0)
dataou_tr = np.array(dataou_tr)
dataou_vl = np.array(dataou_vl)

np.save('./MechanicalToolsClassification_TR_IN.npy', datain_tr)
np.save('./MechanicalToolsClassification_VL_IN.npy', datain_vl)
np.save('./MechanicalToolsClassification_TR_OU.npy', dataou_tr)
np.save('./MechanicalToolsClassification_VL_OU.npy', dataou_vl)





