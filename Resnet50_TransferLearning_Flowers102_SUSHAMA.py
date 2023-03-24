#!/usr/bin/env python
# coding: utf-8

# In[ ]:

'''
CONTEXT:

This python cade has been written for Transfer Learning using pre-trained Resenet50 to accomplish the CV Code Challenge by fellowship.AI
CV Code Challenge requires to use a pre-trained Resnet50 and train it on Oxford Flowers102 dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
In this coding solution the kaggle Flowers-AIPND dataset has been used to obtain a structured dataset such that the total of 8189 flower-images have been divided into 
Train, Valid and Test folders in the approx. ratio 80%, 10% and 10%
The images of flowers are arranged per category or, subfolders(label names) under the three folders - train (6552 images), valid(818 images), test(819 images)
'''
'''' 
ALGORITHM of the Transfer Learning Solution provided in this code:
1. Fetch the total number of different categories\classes of flower images present in the dataset.
2. Download and refer the pre-trained weights of the ResNet50 model: resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
3. Create deep learning model 'model' using Keras Sequential API and add layers to it.
   a) add ResNet50 pre-trained model to the sequential model
   b) add ReLu activation unit to the sequential model
   c) add a dropout layer to the Sequential model, for regularization
   d) add a dense layer with 102 classes and softmax activation function

4. Compile the 'model'
5. Images of flowers in Training data and Validation data are augmented and preprocessed.
6. Batches of augmented images created using ImageDataGenerator
7. EarlyStopping, ModelCheckpoint instances are defined for callbacks to determine actions during model-training
8. Start training the model using fit_generator() function. 
9. Load the weights of the best performing model on the validation set into the model.
10. Analyze the training accuracy and loss, Validation accuracy and loss
11. Repeat the model training with different parameters, different optimizer etc until the desired Validation accuracy is achieved
12. Create Batches of augmented images from Test folder using ImageDataGenerator
13. Evaluate the model on Test dataset generated
'''

'''
FINAL PERFORMANCE METRICS UPON EXECUTION OF THE CODE:
Several epochs of training the model were run with different parameters and optimizers:
 a) Adam optimizer was used and also SGD. SGD didnot give good results. Adam optimizer was retained lately.
 b) Total number of epochs for Training, total number of epochs for Validation and steps per epoch were changed to monitor the effect on Validation Accuracy
 c) Changes to 'steps_per_epoch' were more impactful in enhancing the Validation accuracy
 d) After serevral runs a Validation accuracy of 0.9116 i.e. 91.16% was obtained. There was no overfitting observed when comparing the Training and Validation metrices.
 e) As a final step, evaluation of TEST data (i.e. 819 image files) category-prediction was done. Test accuracy of 0.8937728954467726 obtained.

'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/flowersaipnd" directory.

#print(os.listdir("../input/flowers-recognition/"))
print("Directory structure for the input\parent folder",os.listdir("../input/flowersaipnd/"))


# In[ ]:


labels = os.listdir("../input/flowersaipnd/train/")
print("Directory structure for the training folder called train:\n",labels)
print('Total number of sub-folders',len(labels))


# In[ ]:


num_classes = len(set(labels))
print("Total number of unique labels",num_classes)
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Create model
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print("model creation completed with pre-trained ResNet50\n")
# Do not train first layer (ResNet) as it is already pre-trained
model.layers[0].trainable = False
# Compile model
from tensorflow.python.keras import optimizers

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
print('model has been compiled by calling compile() method')
print('optimizer being used is:',model.optimizer)


# In[ ]:


print("model summary follows:\n")
model.summary()


# In[ ]:


from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

image_size=224
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100
# preprocessing_function is applied on each image but only after re-sizing & augmentation (resize => augment => pre-process)
# Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch) stabilize the inputs to nonlinear activation functions
# Batch Normalization helps in faster convergence
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
# Both train & valid folders must have NUM_CLASSES sub-folders
train_generator = data_generator.flow_from_directory(
        '../input/flowersaipnd/train',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical')
print("data_generator created for Train folder")
validation_generator = data_generator.flow_from_directory(
        '../input/flowersaipnd/valid/',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical') 
print("data_generator created for Valid folder")


# In[ ]:


# Early stopping & checkpointing the best model in ../working dir & restoring that as our model for prediction
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

EARLY_STOP_PATIENCE = 5
cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5',
                                  monitor = 'val_loss',
                                  save_best_only = True,
                                  mode = 'auto')


# In[ ]:


import math

NUM_EPOCHS = 10
fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    validation_data=validation_generator,
    validation_steps=10,
    epochs=NUM_EPOCHS,
    callbacks=[cb_checkpointer, cb_early_stopper])
model.load_weights("../working/best.hdf5")


# In[ ]:


print(fit_history.history.keys())


# In[ ]:


plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history.history['acc'])  
plt.plot(fit_history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()


# In[1]:


#Evaluating the accuracy of TEST dataset classification
test_generator = data_generator.flow_from_directory(
        '../input/flowersaipnd/test/',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_VALIDATION,
        class_mode='categorical')

print("data_generator created for Test folder")
test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))
print('Test accuracy:', test_acc)


# In[ ]:




