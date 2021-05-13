import scipy.misc
import scipy.io as scio
import numpy as np
import os
from sklearn import metrics
import cv2

def data_read(data_path):
    angle = []
    dir=[]
    f= open(data_path)                                 #read steering angles from disk and preprocess
    data = f.read()
    data = data.split()
    for i in data:                                      #if the node end with ".jpg" ignore it. It's for collecting angles
        if i[-1]=='g':
            pass
        else:
            angle.append(float(i) * scipy.pi / 180)     #convert rad.
            ### steering control direction
            if float(i)<-45:
                dir.append(1)
            elif float(i)<45:
                dir.append(2)
            else:
                dir.append(3)
    return angle, dir

def ids_read(test_paths):
    test_ids = []
    for i in test_paths:
        name = i.split(os.path.sep)[-1]
        name = name[:-4]
        test_ids.append(int(name))
    test_ids.sort()
    # test_ids=test_ids[15000:]  #where it start to show
    return test_ids

# calculate dsa threshold
def dsa_thr(dsa,pre_label, tru_label):
    cl=(pre_label!=tru_label)  #misclassification
    fpr, tpr, thresholds = metrics.roc_curve(cl, dsa)
    ### method 1
    # ind=np.argmax((tpr-fpr)/np.sqrt(2))
    # thr=thresholds[ind]
    ## method 2
    dis=fpr**2+(tpr-1)**2
    ind=np.argmin(dis)
    thr=thresholds[ind]
    return thr

### warning pic and dsa values
def cc_detection(idx,image_show,threshold):
    img2 = cv2.imread('images/warning.png')
    img2 = cv2.resize(img2, (455, 256))
    thr=threshold # 0.5129
    dsa = scio.loadmat("data_uncertainty.mat")['dsa']
    # dsa=scio.loadmat("data_uncertainty.mat")['tedsa']

    if dsa[idx] > thr:  ### considering warning
        image_show = cv2.addWeighted(img2, 0.25, image_show, 0.75, 0)
    return image_show