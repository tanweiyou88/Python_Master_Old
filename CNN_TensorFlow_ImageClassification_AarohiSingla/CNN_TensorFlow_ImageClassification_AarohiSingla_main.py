"""
Project: Image Classification using CNN
Youtube tutorial link: https://www.youtube.com/watch?v=JQWCAgwpXbo&list=PLv8Cp2NvcY8DpVcsmOT71kymgMmcr59Mf&index=4&t=4s
Link to download dataset involved in this Youtube tutorial: https://github.com/AarohiSingla/Image-Classification-Using-Convolutional-Neural-Network-and-Tensorflow/tree/main/dataset
Complete step of this project:
1) Ensure the folder structure for dataset is followed: dataset folder -> subfolders (Category/object type/object class as a subfolder name) containing images of object of different categories. Each subfolder contains images of object of the same category.
2) Update the absolute path to the dataset involved
3) [Continue edit]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import csv
import cv2

# The code is compatible with TensorFlow 1.0 but you have TensorFlow 2.0 or higher installed, you can disable the TensorFlow 2.x behavior using the section below:
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


dataset_path = os.path.join('D:\Python_Master\CNN_TensorFlow_ImageClassification_AarohiSingla','rooms_dataset') # The absolute path to the dataset folder containing images of object of different categories
subfolders_list = os.listdir(dataset_path) # Get all subfolder (object categories) names in the dataset folder. "os.listdir()" returns a list of name of all items available in the specified path
print("Categories (types of objects) involved:", subfolders_list)  # Print all object categories involved
print("Number of categories involved:", len(subfolders_list)) # Print the number of object categories invovled. "len()" returns the length in an object/variable

## Data preprocessing
# Gather paths of all files (absolute paths to respective files) in the dataset folder
all_files_path_truth = [] # create an empty variable
for i in range(0,len(subfolders_list),1): # Iterate over each entry of the list. Concept: using for loop -> variable i start with value 0, stop at value = number of items(subfolders) available in its folder, increment by 1 at each step.
    all_files_truth_temp = Path(os.path.join(dataset_path,subfolders_list[i])).glob('*.*') # find(get) all files in the specified directory (in each subfolder, 1 subfolder at each step, in this case) with the specified file extension (every type of file extension in this case)
    for file in all_files_truth_temp: # iterate over each item path (absolute path to each item) obtained. Understand for loop through https://www.geeksforgeeks.org/python-for-loops/
        all_files_path_truth.append(all_files_truth_temp) # Add each item path available in the subfolder in the variable called "all_items_truth". 

# Gather all filenames (item names) in the dataset folder
filenames = [] # create an empty variable
for subfolder in subfolders_list:
    all_filenames = os.listdir(os.path.join(dataset_path , subfolder)) # Iterate over each subfolder of dataset, stores a list of name of all items available in the subfolder 
    # print(all_rooms)
    for filename in all_filenames: # Iterate over each item name available in a subfolder
        filenames.append(filename) # Add each item name available in the subfolder in the variable called "filenames"
        # print(filenames) # Print each item name available in the variable called "filenames". Uncomment this for checking purpose.

# Check if the total number of filenames in the dataset folder gathered same as the total number of item paths available in the dataset folder 
if len(all_files_path_truth) == len(filenames): 
    print('Number of images available in the dataset: '+ str(len(all_files_path_truth)) + '; Number of filenames gathered: '+ str(len(filenames)) + '; All filenames are gathered. Continue data preprocessing part.\n') 

    # Create a list store file information, according to the format (room type, renamed filename)
    category_filepaths_list = []
    for subfolder in subfolders_list:
        all_filenames_redundant = os.listdir(os.path.join(dataset_path , subfolder)) # Iterate over each subfolder of dataset, stores a list of name of all items available in the subfolder 
        for filename_redundant in all_filenames_redundant: # Iterate over each item name available in a subfolder
            category_filepaths_list.append((subfolder, os.path.join(dataset_path, subfolder, filename_redundant))) # renamed each filename according to the format (dataset/room type/original file name)
            # print(rooms_list) # print each entry of the list. Uncomment this for checking purpose.

    # Build a dataframe
    column_labels=['Category','File(Image) Path'] # Specify the column labels
    dataset_df = pd.DataFrame(data=category_filepaths_list, columns=column_labels) # Create Data frame (2D data structure) using the data of "filenames_list", with column labels listed in column_labels.
    print(dataset_df.head()) # Since the parameter of .head() is not specified, print the first 5 rows of the data frame
    print(dataset_df.tail()) # Since the parameter of .tail() is not specified, print the last 5 rows of the data frame
    print('\nNumber of entries in this table (DataFrame):' + str(len(dataset_df)) + '\n') # print the number of entries of this DataFrame

    dataset_df.to_csv(os.path.join('D:\Python_Master\CNN_TensorFlow_ImageClassification_AarohiSingla','DataFrame.csv'),index=True) # write a pandas DataFrame object directly to a CSV file


    category_count = dataset_df['Category'].value_counts() # count the frequency of unique values in the specified column of a DataFrame (EG: Room Type, in this case)
    print('Number of images in each category, in this DataFrame:')
    print(category_count)

    # Resize all images in dataset (becasue every image in the dataset should have the same shape). Then, save all resize images in a variable & label of all the images in another variable
    img_size = 60  # image size, height and width
    images = [] # create an empty variable. This variable will be used to store all resized images
    labels = [] # create an empty variable. This variable will be used to store the label name of the images
    all_filenames = [] # clear the information previously stored in this variable

    for subfolder in subfolders_list:
        subfolder_paths = os.path.join(dataset_path , subfolder) # Iterate over each subfolder of dataset, get the absolute path to each subfolder 
        all_filenames = os.listdir(subfolder_paths) # Get a list of filenames available in each subfolder(category)
        for filename in all_filenames: # Iterate over each filename available in a subfolder
                img = cv2.imread(os.path.join(subfolder_paths,filename)) # Read that image
                img = cv2.resize(img,(img_size,img_size)) # Resize that image 
                images.append(img) # Add each resized image in the variable called "images"
                labels.append(subfolder) # Add each image label (category of the image as its laabel) in the variable called "labels"

    images = np.array(images) # Transform the image array to a numpy type (Convert every image into an array). Because computer understand an image in the form of array of pixels [the pixel value range depends on the image type]
    images = images.astype('float32')/255.0 # Normalize pixel value in each colour by dividing with the maximum pixel value of the channel (EG: 255)
    print(images.shape) # check the size of the images in the variable "images". (Number of images in the variable, pixel numbers (size) in height, pixel numbers (size) in width, colour channel)

    # convert(encode) each label name from a string into an integer so that the machine learning model can understand the label
    category_labels=dataset_df['Category'].values # get the values under the column called "category"
    labelencoder = LabelEncoder()
    category_label_encoded = labelencoder.fit_transform(category_labels) # encode target labels with value between 0 and (number of category - 1)
    print(category_label_encoded)

    # Convert the labels from scalar into OneHotEncoder form. becasue deep learning only accept the input in the form of OneHotEncoder.
    category_label_encoded = category_label_encoded.reshape(-1,1)
    print(category_label_encoded)

    onehotencoder = OneHotEncoder()
    category_label_encoded_OneHot = onehotencoder.fit_transform(category_label_encoded)
    print(category_label_encoded_OneHot) 
    print(category_label_encoded_OneHot.shape) # print the dimension -> (number of rows, number of columns). The number of columns here depends of the number of categories

## Organizing dataset structure (Split dataset into train & test sets). No val set in this project

    images, category_label_encoded_OneHot = shuffle(images, category_label_encoded_OneHot, random_state=1) # random_state=value controls how the input data is shuffled, refering to the (value)th combination of the random shuffle input data (EG: dataset here), such that the results is reproducable when the script is executed every time. https://medium.com/analytics-vidhya/what-is-random-state-in-machine-learning-84c1c9dffaad
    train_x, test_x, train_y, test_y = train_test_split(images, category_label_encoded_OneHot, test_size=0.05, random_state=415) # In machine learning context, x refers to image, y refers to label of the corresponding image. test_size = 0.05 means 5% of the dataset is split into the test set, after the shuffling completed

    # inspect the shape of the training and testing set
    print('Shape of train_x variable -> (Number of images in training set, number of pixels in height,  number of pixels in width, colour channel):\n', train_x.shape) # (Number of images in training set, number of pixels in height,  number of pixels in width, colour channel)
    print('Shape of train_y variable -> (Number of image labels in training set, number of categories):\n', train_y.shape) # (Number of image labels in training set, number of categories)
    print('Shape of test_x variable -> (Number of images in test set, number of pixels in height,  number of pixels in width, colour channel):\n', test_x.shape) # (Number of images in test set, number of pixels in height,  number of pixels in width, colour channel)
    print('Shape of test_y variable -> (Number of image labels in test set, number of categories):\n', test_y.shape) # (Number of image labels in test set, number of categories)

    ## Develop CNN model architecture with TensorFlow 
    num_classes = len(subfolders_list) # Number of classes/categories involved
    flattened_image_size = images.shape[1]*images.shape[2]*images.shape[3] # Flattening an image means combining multiple layers (colour channels) into 1 layer. Since each image in the data set must have same dimension/size and colour channel. 1 flattened image has the size = (number of pixels in height) * (number of pixels in width) * (number of colour channels) 
    
    # architecture hyperparameter
    learning_rate = 0.001
    training_iters = 10 # number of epoch
    batch_size = 16
    display_step = 20 # just for display purpose, not important for learning purpose. Can comment this line.
    num_channels = images.shape[3]

    x_placeholder = tf.placeholder(tf.float32, shape=[None,img_size,img_size,num_channels]) # Create a variable to store images in the form of float.32, with specified shape parameters. shape = [batch size,img_size,img_size,num_channels]
    y_placeholder = tf.placeholder(tf.float32, shape=[None,num_classes]) # Create a variable to store label corresponding to images stored in "c" in the form of float.32, with specified number of classes. shape = [batch size,num_classes]

    print('Shape of placeholder', x_placeholder.shape, y_placeholder.shape)
    
    # CNN learn features on training set, then find the learned features on test set
    # CNN part
    def conv2d_define(x,W,b,strides=1):  # Call this function to get 1 convolutional layer. This self-defined a definition to get all parameters required for convolutional 2D layer & ReLU activation function before calling the convolutional 2D layer & ReLU activation function from the neural network part of TensorFlow. x is the input image, W is the weight on the filter of the layer (weight), b is bias [something to add on]
        # neuron/perceptron has 2 parts: summation & activation parts
        # summation part
        x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding='SAME') # Convolutional 2D layer. summation = x*W
        x=tf.nn.bias_add(x,b) # summation = x*W + b
        return tf.nn.relu(x) # ReLU layer. tf.nn.relu(x) is the activation part, using ReLU, to replace all negative avalues with 0   
    
    def maxpool2d_define(x,k=2): # Call this layer to get 1 max pooling layer. This self-defined a definition to get all parameters required for max pooling layer before calling the max pooling layer function from the neural network part of TensorFlow. k means window size of the filter
        return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME') # Pooling layer. tf.nn.max_pool() reduce the size of image
    
    weights ={ # create a dictionary called weight, stores the value of weight for each layer.
        # structure: key : value of the key
        'w1': tf.Variable(tf.random_normal([5,5,3,32]),name='w1'), # At the first layer of this CNN model (convolutional layer), apply 32 filters of window size 5x5 with 3 colour channels on the input image. The input image shape in this project is 60x60x3
        'w2': tf.Variable(tf.random_normal([5,5,32,64]),name='w2'), # since the output of the first layer will be the input to this second layer, tf.random_normal([height of filter window, width of filter window, Number of images output by the previous layer, Number of filters with window size mentioned at the first 2 parameters which will be apply on the images output on the previous layer])  
        'w3': tf.Variable(tf.random_normal([5,5,64,128]),name='w3'),
        'wd1': tf.Variable(tf.random_normal([8 * 8 * 128, 2048]),name='wd1'), # This is the fully connected layer. Convert 2D images into 1D array (flatten the image). 2048 refers to the number of neurons on this layer. Each neuron will pass through each 1D image of size = 8 * 8 * 128, where 8 * 8 is the window size of its filter, 128 is the number of images output by previous layer
        'wout': tf.Variable(tf.random_normal([2048, num_classes]),name='wout') # This is the final(output) layer, which consists of number of neurons = number of class. Since the output of previous layer will be the input of next layer, so 2048 images output by previous layer will feed into this layer. 
    }

    biases ={ # create a dictionary called biases, stores the value of bias for each layer.
        'b1': tf.Variable(tf.random_normal([32]),name='b1'), # At the first layer, the biases is 32
        'b2': tf.Variable(tf.random_normal([64]),name='b2'), # At the second layer, the biases is 64
        'b3': tf.Variable(tf.random_normal([128]),name='b3'), # At the third layer, the biases is 128
        'bd1': tf.Variable(tf.random_normal([2048]),name='bd1'), # At the fourth layer (fully connected layer), the biases is 2048 
        'bout': tf.Variable(tf.random_normal([num_classes]),name='bout'), # At the final(output) layer, the biases equals to the number of classes
    }

    def conv_net_define(x,weights,biases): # define the whole CNN network
        # reshape input to 60x60x3 size
        x = tf.reshape(x,shape=[-1,60,60,3])

        
        print('#######################################################')
        print('size of x is ', x.shape)

        # input is an image with 30 * 30 * 3
        # 1st convolutional layer
        conv1 = conv2d_define(x,weights['w1'],biases['b1'])
        conv1 = maxpool2d_define(conv1,k=2)
        print('#######################################################')
        print('size after 1st conv layer and pooling layer is ', conv1.shape)

        # input is 30 * 30 * 32 images
        # 2nd convolutional layer
        conv2 = conv2d_define(conv1,weights['w2'],biases['b2'])
        conv2 = maxpool2d_define(conv2,k=2)
        print('#######################################################')
        print('size after 2nd conv layer and pooling layer is ', conv2.shape)

        # input is 15 * 15 * 64 images
        # 3rd convolutional layer
        conv3 = conv2d_define(conv2,weights['w3'],biases['b3'])
        conv3 = maxpool2d_define(conv3,k=2)
        print('#######################################################')
        print('size after 3rd conv layer and pooling layer is ', conv3.shape)

        # input is 8 * 8 * 128 images
        # Fully connected layer
        # Reshape conv3 output to fit fully connected layer input = 8 * 8 * 128 = 8192
        fc1 = tf.reshape(conv3,[-1,weights['wd1'].get_shape().as_list()[0]])
        print('#######################################################')
        print('size after flattening the image ', fc1)

        fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
        dc1=tf.nn.relu(fc1)
        print('#######################################################')
        print('shape after fully connected layer ', fc1)
        
        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term.  
        out=tf.add(tf.matmul(fc1,weights['wout']),biases['bout'])
        print('#######################################################')
        print('Output layer ')
        return out

    # create the whole CNN network
    model = conv_net_define(x_placeholder,weights,biases)
    print(model) # check the whole CNN network

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=y_placeholder)) # calculate the loss, the difference between the actual and predicted y value (labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # optimizer will update the weight of the model based on the loss calculated

    # initializing the variables
    init=tf.global_variables_initializer()
    cost_history=[]
    num_epochs = 10
    # the execution
    sess = tf.Session()
    sess.run(init)

    train_y=train_y.todense()
    
    for i in range(num_epochs):
       a, c = sess.run([optimizer,cost],feed_dict={x_placeholder: train_x, y_placeholder: train_y}) # working. In this for loop, we input feed_dict, which contains the training data, to train the model
       cost_history=np.append(cost_history,c) # calculate the loss at each epoch
       print('epoch :', i, '-', 'cost:', c)

    test_y=test_y.todense()

    correct_prediction = tf.equal(tf.argmax(model,1),tf.argmax(y_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print('Accuracy: ',sess.run(accuracy,feed_dict={x_placeholder: test_x, y_placeholder: test_y}))

    # print('Correct Prediction: ', str(correct_prediction))
    # print('\nAccuracy: ', str(accuracy))
       
    # print(cor
    # uracy)

else: 
    print('Number of images available in the dataset: '+ str(len(all_files_path_truth)) + '; Number of filenames gathered: '+ str(len(filenames)) +'; Not all filenames are gathered. Please recheck and ensure all filenames are gathered before continuing data preprocessing part.\n')




