# Resnet50_TransferLearning_Flowers102_SUSHAMA
CONTEXT:

This python code has been written for Transfer Learning using pre-trained Resenet50 to accomplish the Computer Vision Code Challenge by fellowship.AI
CV Code Challenge requires to use a pre-trained Resnet50 and train it on Oxford Flowers102 dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
In this coding solution the kaggle Flowers-AIPND dataset has been used to obtain a structured dataset such that the total of 8189 flower-images have been divided into 
Train, Valid and Test folders in the approx. ratio 80%, 10% and 10%, respectively.
The images of flowers are arranged per category or, subfolders(label names) under the three folders - train (6552 images), valid(818 images), test(819 images)
 
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


FINAL PERFORMANCE METRICS UPON EXECUTION OF THE CODE:
Several epochs of training the model were run with different parameters and optimizers:
 a) Adam optimizer was used and also SGD. SGD didnot give good results. Adam optimizer was retained lately.
 b) Total number of epochs for Training, total number of epochs for Validation and steps per epoch were changed to monitor the effect on Validation Accuracy
 c) Changes to 'steps_per_epoch' were more impactful in enhancing the Validation accuracy
 d) After serevral runs a Validation accuracy of 0.9116 i.e. 91.16% was obtained. There was no overfitting observed when comparing the Training and Validation metrices.
 e) As a final step, evaluation of TEST data (i.e. 819 image files) category-prediction was done. Test accuracy of 0.8937728954467726 obtained.

