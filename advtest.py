import cv2
import numpy as np
from imutils import paths
import os

def light_change(img):
    #  adjust brightness
    rows,cols,channel=img.shape
    a = np.random.rand(1)
    b = np.random.randint(0,200,1)
    dst = img.copy()
    dst = dst * a + b
    dst[dst>255]=255

    return dst

def add_occlusion(img):

    row=np.random.randint(0,236,1)[0]
    col=np.random.randint(0,415,1)[0]
    noise=np.random.random([20,40,3])
    dst=img.copy()
    dst[row:row+20,col:col+40,:]=noise*100+dst[row:row+20,col:col+40,:]
    dst[dst>255]=255
    return dst

def add_noise(img):
    noise = np.random.random([256, 455, 3])
    b=np.random.randint(0,100,1)
    dst=img.copy()+noise*b
    dst[dst > 255] = 255
    return dst

test_ids=[]

test_paths = list(paths.list_images(os.getcwd()+"/test"))
#get test images ids (names)
for i in test_paths:
    name = i.split(os.path.sep)[-1]
    name = name[:-4]
    test_ids.append(int(name))
test_ids.sort()

for i in test_ids:
    image = cv2.imread(os.getcwd()+"/test/"+str(i)+".jpg")   #read images from disk
    img1=light_change(image.copy())
    path1=os.getcwd() + "/test1/" + str(i) + ".jpg"
    cv2.imwrite(path1, img1)
    img2=add_occlusion(image.copy())
    path2=os.getcwd() + "/test2/" + str(i) + ".jpg"
    cv2.imwrite(path2, img2)
    img3=add_noise(image.copy())
    path3=os.getcwd() + "/test3/" + str(i) + ".jpg"
    cv2.imwrite(path3, img3)