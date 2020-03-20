


#Import Functions

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D,concatenate, BatchNormalization, Dropout, Flatten, Activation
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
import keras.models as model
from PIL import ImageFile




# Set this to the current working directory
os.chdir("/home/ashwani/Deep_learning_Scripts")

if not os.path.isdir("Weights"):
    os.mkdir("Weights")

# Set the inital parameters according the needs
# Generally Image size of 224, 224 is goood enough,
# We can set up the other parameters according to the training needs
batch_size = 3
img_width = 224
img_height = 224
epochs = 10
learn_rate = 1e-5
# Number of classes in the problem statement
nclasses =  7

# Setting up[ the numbers of frozen layers
# frezzed layers wont be trained and hence we can make use of pre learned model.

layers_frozen=0
# Path for saving the results

model_path = './'

# setting the train data path
train_data_dir ="/home/ashwani/Deep_learning_Scripts/Train"
# validation data path
validation_data_dir = "/home/ashwani/Deep_learning_Scripts/Validate"

# Setting up name for logs
Name = "InceptionResnetV2".format(int(time.time())) 



#Image-PreProcessing
# These steps creates train and validation_datagen for tarining and validation data
# Parameters involves the parameters for pre processing the images
# In other words augmenting images for better results
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,width_shift_range=0.3,
                                   height_shift_range=0.3,rotation_range=30,shear_range=0.5,zoom_range=.7,
                                   channel_shift_range=0.3,cval=0.5,vertical_flip=True,fill_mode='nearest')
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(
    img_height, img_width), batch_size=batch_size, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(
    img_height, img_width), batch_size=batch_size, class_mode='categorical')

# Setting up the train steps and val steps on the basis of length of data gens
# These params will be used while model training

train_steps = train_generator.__len__()
val_steps = validation_generator.__len__()




# Selecting the pretrained model for transfer learning
# Generally Nasnet is slowest and most accurate
#  We can select the architecture depending upon the processing power.
# For general purposes Resnet50 will suffice.

architecture=3

if architecture==1:
    base_model = InceptionResNetV2(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="InceptionResNetV2"
elif architecture==2:
    base_model = DenseNet121(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="DenseNet121"
elif architecture==3:
    base_model = ResNet50(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="ResNet50"
elif architecture==4:
    base_model = NASNetMobile(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="NASNetMobile"
elif architecture==5:
    base_model = MobileNet(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="MobileNet"
elif architecture==6:
    base_model = InceptionV3(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="InceptionV3"
else:
    print ("Wrong Architecture Input")
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nclasses, activation='softmax')(x)
pmodel = Model(base_model.input, predictions)



# Setting up the model

model = pmodel
for layer in model.layers[:-layers_frozen]:
    layer.trainable = False

nadam = Nadam(lr=learn_rate)

model.compile(optimizer=nadam, loss='categorical_crossentropy',metrics=['accuracy'])
print('=> done building model <=')





# Storing logs of learning using Tensorboard

tensorboard = TensorBoard(
    log_dir='./logs'.format(Name), histogram_freq=0, write_graph=True, write_images=False)

# File path for storing model weights

filepath=os.path.join(
                                os.path.abspath(model_path), 'Weights/top_model_weights_'+'Frozen_Layers.h5')



# Setting up the checkpoints for model
# checkpoints helps in storing the best trained model only
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

print('=> created callback objects <=')
print('=> initializing training loop <=')

# Running the model
history = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=epochs,
                              validation_data=validation_generator, validation_steps=val_steps,
                              workers=2, 
                              use_multiprocessing=False, 
                              max_queue_size=500, 
                              callbacks=callbacks_list)




filepath=os.path.join(
                                os.path.abspath(model_path), 'Weights/top_model_weights_'+'Frozen_Layers.h5')

print('=> loading best weights <=')
model.load_weights(filepath)
print('=> saving final model <=')
Final_Weights='Weights/model_'+architecture_name+"_"+str(layers_frozen)+'Frozen_Layers.h5'
pmodel.save(os.path.join(os.path.abspath(model_path), Final_Weights))





# These steps are only for checking predictions and only needed 
# to run if you need to check predictions on your data


# Loading the model with best weights

new_model=tf.keras.models.load_model(filepath)
#new_model.summary()



#Predictions


predictions=[]

# Set up this to the validation images path
img_path= "/home/ashwani/Deep_learning_Scripts/Validate/"  #Set This To The Val Directory Path

# Path to csv file that will store the results

CSV_Name="/home/ashwani/Deep_learning_Scripts/Validate.csv"    #Set CSV Name To Be Generated
filenames=validation_generator.filenames
for i in filenames:
    img = tf.keras.preprocessing.image.load_img(img_path+i, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img) 
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y=new_model.predict(x)
    predict=str(y.argmax(axis=-1))
    predict=predict.replace("[","")
    predict=predict.replace("]","")
    predict=int(predict)
    predictions.append(predict)
labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
labels[4]='None'
predicted = [labels[k] for k in predictions]
results=pd.DataFrame({"Filename":filenames,"Predictions":predicted})
actual=[]
for i in results['Filename']:
    head, sep, tail = i.partition('/')
    actual.append(head)
results['Actual']=actual
results.to_csv(CSV_Name)






