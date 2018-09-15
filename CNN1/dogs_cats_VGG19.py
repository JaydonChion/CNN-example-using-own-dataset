import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K  
K.set_image_dim_ordering('tf')   #we use tensorflow as backend, can consider other option like theano or CNTK
K.set_image_data_format('channels_last')   #We set the format of our input image in this way : (length, width,channel)
from keras.models import Sequential   
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.applications import VGG19    #We use pretrained network to save our time and increase accuracy
from keras.layers import Dropout        
from keras.models import Model
from keras.callbacks import EarlyStopping  #stop the training early if the accuracy fail to improve after some time
from sklearn.metrics import accuracy_score
import pandas as pd
import time


datagen_train = ImageDataGenerator(#read images from train data directory
    rescale = 1./255,              #some data augmentation
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    channel_shift_range = np.random.random_sample()
)

datagen_val = ImageDataGenerator(  
    rescale = 1./255
)


train_data = datagen_train.flow_from_directory(

    directory = '/Users/jetherngchion/Downloads/all-2/train_set',  #path to train data directory 
    target_size = (100,100),
    class_mode = 'binary', #change to categorical if you have more than 2 classes
    shuffle = True,
    batch_size = 50

)


val_data = datagen_val.flow_from_directory(

    directory = '/Users/jetherngchion/Downloads/all-2/validation_set',
    target_size = (100,100),
    class_mode = 'binary',  #change to categorical if you have more than 2 classes
    shuffle = True,
    batch_size = 50

)


print('class indices : {}'.format(train_data.class_indices))   #show your data classes


base_model = VGG19(include_top = False, weights = 'imagenet', input_shape = (100, 100, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x) #new FC layer, random init
x = Dropout(0.4)(x)
predictions = Dense(1, activation='softmax')(x)  
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:  #freeze the existing pretrained layers so that our training wont affect their weights
    layer.trainable = False
model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy']) #there are other kind of losses, Adam optimizer is a good start, can consider RMSprop or momentum optimizer
print("{:<10} Pretrained model layers: {}".format('[INFO]', len(base_model.layers)))
print("{:<10} Total number of layers : {}".format('[INFO]', len(model.layers)))


model.summary() #have a look at the layers


#------------------------------------------------------------------------------
                                #Training begin
#------------------------------------------------------------------------------

train_start_time = time.time()


model.fit_generator(
    train_data,
    epochs = 25,
    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)],
    validation_data = val_data)
model.save('/Users/jetherngchion/Downloads/all-2/cnn_model_VGG19.h5')

print("It takes {:.2f} min to train the model".format((time.time() - train_start_time)/60 ))

scores = model.predict_generator(val_data)
print(scores)

y_pred = [round(score[0]) for score in scores] #round the score, above 0.5 is dog, below is cat
print(y_pred)


y_true = [0 if 'cat' in filename[:3] else 1 for filename in os.listdir('/Users/jetherngchion/Downloads/all-2/test_set')]

print(y_true)

print(accuracy_score(y_true, y_pred))

pd.crosstab(pd.Series(y_true), pd.Series(y_pred), rownames=['True'], colnames=['Pred'], margins=True)


