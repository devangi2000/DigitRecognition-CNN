import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import pickle

#path to the folder containing digits from 0 through 9 in separate folders
path= 'myData'

mylist = os.listdir(path)
print('Total number of classes detected: ', len(mylist))
noOfClasses = len(mylist)
images = []
classNo = []
test_ratio = 0.2
valRatio = 0.2
imageDimensions = (32,32,3)

batchSizeVal = 32
epochsValue = 2
stepsPerEpoch = 2000

print('Importing classes')
#get the name of each image inside each folder
for x in range(noOfClasses):
    #this goes to myData/0/img001-00001 and so on, listing out each item
    myPicList = os.listdir(path+'/'+str(x))
    #store each item in a list
    for y in myPicList:
        #read every image
        curImg = cv2.imread(path+'/'+str(x)+'/'+str(y))
        #presently, images are (180x180) which is computationally expensive
        #therefore, I'm resizing it to 32x32
        curImg = cv2.resize(curImg, (imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
        # so we now have a corresponding class number associated with every image in the image list where all IDs are stored
    print(x, end=' ')
    # 0 1 2 3 4 5 6 7 8 9
print('')
#convert images and classes to numpy arrays
images = np.array(images)
classNo = np.array(classNo)

print(len(classNo)) #10160
print(len(images))  #10160

print(images.shape) #(10160, 32, 32, 3)
print(classNo.shape) #(10160,)

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)
print(X_train.shape) #(6502, 32, 32, 3)
print(X_validation.shape) #(1626, 32, 32, 3)
print(X_test.shape) #(2032, 32, 32, 3)

numOfSamples = []
#for each class, this gives us how many images we have
for x in range(noOfClasses):
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)
#[621, 662, 660, 660, 651, 645, 656, 650, 643, 654]
plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses), numOfSamples)
plt.title('Number of images of each class')
plt.xlabel('Class ID')
plt.ylabel('Number of images')
plt.show()

def preProcessing(img):
    #convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # makes the lighting even
    img = cv2.equalizeHist(img)
    # normalization converts the values ranging from 0 through 255 to values bwtween 0 and 1
    img = img/255
    return img

#img = preProcessing(X_train[30])
# we resize the image to be able to see it clearly
#img = cv2.resize(img, (300,300))
#cv2.imshow('Preprocessed Image', img)
#if we don't use the waitKey, it closes
#cv2.waitKey(0)

print('Shape before Preprocessing: ', X_train[30].shape)

#(32, 32, 3)
#picks each image one by one from X_train and passes it through the preprocessing function
#it is also necessary to convert this list inot a numpy array

X_train = np.array(list(map(preProcessing, X_train)))
X_validation = np.array(list(map(preProcessing, X_validation)))
X_test = np.array(list(map(preProcessing, X_test)))

img = cv2.resize(X_train[30], (300,300))
cv2.imshow('Preprocessed Image', img)
#if we don't use the waitKey, it closes
cv2.waitKey(0)
print('Shape after preprocessing: ', X_train[30].shape)
# (32, 32)

#we add a depth of 1 for the cnn to be able to run properly
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

#so the first 3 parameters remain the same
print('Shape of training data after adding a depth of 1: ', X_train.shape)
print('Shape of test data after adding a depth of 1: ', X_test.shape)
print('Shape of validation data after adding a depth of 1: ', X_validation.shape)

#Data Augmentation (done by ImageDataGenerator)
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
#help image generator calculate certain statistics before performing transformations
dataGen.fit(X_train)
#this does not generate images before starting the training
#instead, we generate them as we go along


#Converts a class vector (integers) to binary class matrix.
y_train = to_categorical(y_train, num_classes=noOfClasses)
y_test = to_categorical(y_test, num_classes=noOfClasses)
y_validation = to_categorical(y_validation, num_classes=noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1),
                      activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

model = myModel()
print(model.summary())

#Now we request the images in batches
#for tf version 2.0.0, use model.fit_generator
history = model.fit(dataGen.flow(X_train, y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpoch,
                                 epochs=epochsValue,
                                 validation_data=(X_validation, y_validation),
                                 shuffle=1)
#history variable stores the data
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')

plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score achieved = ', score[0])
print('Test accuracy achieved = ', score[1])

#save this file and use it in our testing model with webcam or using images
# wb means write bytes
pickle_out = open('A/model_trained.p', 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()
#this creates the pickle object and saved it in the directory
