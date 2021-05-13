############################ steering direction classification
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
import tensorflow.keras.utils as np_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import cv2
from keras import backend as K
from utils import *
from dif_dsa import *

import scipy.io as scio

# # with a Sequential model
def hidden_outs(model, train_ids):
    output=np.array([])
    layer_out = K.function([model.layers[0].input],
                           [model.layers[10].output])  # get_10th_layer_output=last dense output
    for i in train_ids:
        image = cv2.imread(os.getcwd()+"/train/"+str(i)+".jpg")   #read images from disk
        image=cv2.resize(image[-150:], (200,66))
        image = img_to_array(image)/255
        temp_o = layer_out(image[None])[0]
        if i==train_ids[0]:
            output=temp_o
        else:
            output = np.concatenate((output, temp_o), axis=0)
    return output

# ################# dsa calculation
def dsa_cal(train_paths,test_paths,model,dirmodel,y_train):
    # trainpaths = list(paths.list_images(os.getcwd()+"/train"))
    # testpaths = list(paths.list_images(os.getcwd()+"/test"))
    train_ids=ids_read(train_paths)
    test_ids=ids_read(test_paths)
    x_train=hidden_outs(model,train_ids)
    x_test=hidden_outs(model,test_ids)
    train_ats=x_train
    test_ats=x_test
    # y_train=label
    pre_train=dirmodel.predict_classes(x_train)
    pre_test=dirmodel.predict_classes(x_test)
    class_matrix, all_idx = cal_cla_matrix(y_train)
    # dsa = cal_dsa0(train_ats, y_train, test_ats, pre_test, class_matrix,all_idx)
    dsa_tr = cal_dsa3(train_ats, y_train, train_ats, pre_train, class_matrix)
    dsa_te = cal_dsa3(train_ats, y_train, test_ats, pre_test, class_matrix)
    # mdic={'trdsa':dsa_tr,'tedsa':dsa_te}
    # scio.savemat('data_uncertainty.mat',mdic)
    return dsa_tr, dsa_te, pre_train, pre_test

def dirModel():
    model = Sequential()
    model.add(Dense(3,activation='elu'))
    return model

if __name__ == "__main__":
    model=load_model("model/model.h5")
    data_path="data.txt"
    angle,dir=data_read(data_path)
    train_paths = list(paths.list_images(os.getcwd()+"/train"))
    train_ids=ids_read(train_paths)

    x_train=hidden_outs(model,train_ids)
    label=np.array(dir)[:40000]-1
    # Generators
    y_train = np_utils.to_categorical(label, 3)
    #
    dirmodel=dirModel()
    dirmodel.add(Activation('softmax'))
    opt=tf.keras.optimizers.Adam(lr=0.0002)
    dirmodel.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    dirmodel.fit(x_train[:40000],
                 y_train[:40000],
                 epochs=10,
                 batch_size=128,
                 shuffle=True,
                        # validation_data=validation_generator
                  )

    dirmodel.save('model/dirmodel.h5')

