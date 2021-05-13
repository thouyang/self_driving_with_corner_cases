#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
from tensorflow.keras.models import load_model
import time
from utils import *
#------------------------------------------------------
#This section for reduce some errors related to GPU allocation on my system.
#it may not neccesary for yours. If it is, removing this part may increase the performance.
# from tensorflow import Session,ConfigProto
# from keras.backend.tensorflow_backend import set_session
from tensorflow import Session, ConfigProto  # for tensorflow>1.5 using .compat.v1
from tensorflow.python.keras.backend import set_session

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(Session(config=config))
#--------------------------------------------------------

input_shape = (66, 200, 3)
smooth_angle=0
cc_warning=True # selecting if considering corner case warning

data_path="data.txt"
angle,dir=data_read(data_path)
test_paths = list(paths.list_images(os.getcwd()+"/test"))
test_ids=ids_read(test_paths)
model = load_model("model/model.h5")                      #import our model

sahin_direksiyon = cv2.imread("images/sahin_direksiyon_simiti.png") #read steering image
sahin_konsol = cv2.imread("images/sahin_on_konsol_2.jpg") #frontside image of a car.

#resize it
sahin_direksiyon=cv2.resize(sahin_direksiyon, (270,210))
#get it's shape
rows,cols,level = sahin_direksiyon.shape

for i in test_ids:
    image = cv2.imread(os.getcwd()+"/test/"+str(i)+".jpg")   #read images from disk
    image_show = image
    image=cv2.resize(image[-150:], (200,66))                
    image = img_to_array(image)/255                   #convert to numpy array
    result = -model.predict(image[None])*180.0/scipy.pi             #make a prediction
    print("Actual Angle= {} Predicted Angle= {}".format(str(angle[i]),str(-result)))

    ### show
    if cc_warning:
        image_show=cc_detection(i,image_show,0.5129) # with corner case detection
    cv2.imshow("Self Driving Car",cv2.resize(image_show,(800,398)))     #show image 

    #this section just for the smoother rotation of streeing wheel.
    smooth_angle += 0.2 * pow(abs((result - smooth_angle)), 2.0/3.0)*(result - smooth_angle)/abs(result-smooth_angle)
    M=cv2.getRotationMatrix2D((cols/2,rows/2),smooth_angle,1)
    dst= cv2.warpAffine(sahin_direksiyon, M, (cols,rows))
    #sahin_konsol[20:230,30:300]=dst #If you want to show frontside of the car just use this and replace dst with sahin_konsol in the next line. Note: Optional it's just for fun.

    cv2.imshow("Sahin Wheel",dst)
    #small delay for Optimus Prime level computers
    time.sleep(0.02)        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()





