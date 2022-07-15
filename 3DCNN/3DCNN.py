import os
import numpy as np
import cv2
import argparse
import locale
import matplotlib.pyplot as plt
import tensorflow.keras
import h5py
import pickle as pk
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv3D, Flatten, MaxPooling3D, Dropout, BatchNormalization, AveragePooling3D, GlobalAveragePooling3D, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# https://pyimagesearch.com/2019/01/28/keras-regression-and-cnns/
# different inputs: https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

#Helpers functions

# already in gray scale, nothing to be done
def gray_scale(input):
	return input

#input-Array with shape (N, H, W)
def gradient_x(input):
	result = np.gradient(np.array(input, dtype=float), axis=1)
	return result

#input-Array with shape (N, H, W)
def gradient_y(input):
	result = np.gradient(np.array(input, dtype=float), axis=2)
	return result

def optflow_x(input):
	shape = input.shape
	newshape = (shape[0], shape[1], shape[2], (shape[3]-1), shape[4])
	result = np.ndarray(shape=newshape, dtype=float)
	#source https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
	#loop over all samples:
	for i in range(0, input.shape[0]):
		#loop over all frames in one cube:
		for k in range(0, input.shape[3]-1):
			prev = input[i, :, :, k, 0]
			next = input[i, :, :, k+1, 0]
			# default params
			flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			x = flow[:, :, 0]
			result[i, :, :, k, 0] = x
	return result

def optflow_y(input):
	shape = input.shape
	newshape = (shape[0], shape[1], shape[2], (shape[3]-1), shape[4])
	result = np.ndarray(shape=newshape, dtype=float)
	#loop over all samples:
	for i in range(0, input.shape[0]):
		#loop over all frames in one cube:
		for k in range(0, input.shape[3]-1):
			prev = input[i, :, :, k, 0]
			next = input[i, :, :, k+1, 0]
			# default params
			flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			x = flow[:, :, 1]
			result[i, :, :, k, 0] = x
	return result

def create_branch_model(input):
	#branching into 2, conv3D(7,7,3) --> pooling(2,2,1)
	a, b = conv_2branches(input)

	#branching into 3, conv3D(7,6,3) --> pooling(3,3,1) --> conv3D(7,4,1)
	c, d, e = conv_3branches(a)
	f, g, h = conv_3branches(b)

	#merge branches and flatten them
	combined = concatenate([c, d, e, f, g, h,])
	output = Flatten()(combined)

	model = Model(inputs=input, outputs=output)
	return model

def conv_2branches(input_tensor):
	a = Conv3D(filters=1, kernel_size=(7, 7, 3), padding="valid", activation="relu", data_format='channels_last')(input_tensor)
	a = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding="valid")(a)

	b = Conv3D(filters=1, kernel_size=(7, 7, 3), padding="valid", activation="relu", data_format='channels_last')(input_tensor)
	b = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding="valid")(b)
	return a, b

def conv_3branches(input_tensor):
	c = Conv3D(filters=1, kernel_size=(7, 6, 3), padding="valid", activation="relu")(input_tensor)
	d = Conv3D(filters=1, kernel_size=(7, 6, 3), padding="valid", activation="relu")(input_tensor)
	e = Conv3D(filters=1, kernel_size=(7, 6, 3), padding="valid", activation="relu")(input_tensor)

	c = AveragePooling3D(pool_size=(3, 3, 1), strides=None, padding="valid")(c)
	d = AveragePooling3D(pool_size=(3, 3, 1), strides=None, padding="valid")(d)
	e = AveragePooling3D(pool_size=(3, 3, 1), strides=None, padding="valid")(e)

	c = Conv3D(filters=1, kernel_size=(7, 4, 1), padding="valid", activation="relu")(c)
	d = Conv3D(filters=1, kernel_size=(7, 4, 1), padding="valid", activation="relu")(d)
	e = Conv3D(filters=1, kernel_size=(7, 4, 1), padding="valid", activation="relu")(e)
	return c, d, e

def create_hardwired_input(xtrain, xtest):
	trainA = gray_scale(xtrain)
	testA = gray_scale(xtest)

	trainB = gradient_x(xtrain)
	testB = gradient_x(xtest)

	trainC = gradient_y(xtrain)
	testC = gradient_y(xtest)

	trainD = optflow_x(xtrain)
	testD = optflow_x(xtest)

	trainE = optflow_y(xtrain)
	testE = optflow_y(xtest)

	return trainA, trainB, trainC, trainD, trainE, testA, testB, testC, testD, testE


####################EXECUTION#######################

#run on gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(f"GPU available: " + str(tensorflow.test.is_built_with_cuda()))

# load data
image_data = np.load("/home/rueskamp/tmp/pycharm_project_123/dataset_36.npy")
labels = np.load("/home/rueskamp/tmp/pycharm_project_123/fake_test_label_36.npy")

oldshape = image_data.shape
newshape = (oldshape[0], oldshape[1], oldshape[2], oldshape[3], 1)
#reshaping to add one more dimension
image_data = image_data.reshape(newshape)

#splitting data into training and test data
xtrain, xtest, ytrain, ytest=train_test_split(image_data, labels, test_size=0.15)

#creating multiple inputs as suggested in Ji 2010 (grayscale, gradientx, gradienty, optflowx optflowy)
trainA, trainB, trainC, trainD, trainE, testA, testB, testC, testD, testE = create_hardwired_input(xtrain, xtest)

######BUILDING MODEL##############

# 2 different inputShapes
inputShape1 = (60, 40, 7, 1) # grayscale and gradient x, y
inputShape2 = (60, 40, 6, 1) # optflow

# Inputs for each hardwired version (A-E)
inputA = Input(shape=inputShape1)
inputB = Input(shape=inputShape1)
inputC = Input(shape=inputShape1)
inputD = Input(shape=inputShape2)
inputE = Input(shape=inputShape2)

modelA = create_branch_model(inputA)
modelB = create_branch_model(inputB)
modelC = create_branch_model(inputC)
modelD = create_branch_model(inputD)
modelE = create_branch_model(inputE)

# combine output of the five branches
combined = concatenate([modelA.output, modelB.output, modelC.output, modelD.output, modelE.output])

# apply a FC layer and a regression prediction on the combined outputs
z = Dense(128, activation="relu")(combined)
output = Dense(1, activation="linear")(z)

# our model will accept the inputs of the 5 branches and then output a single value
model = Model(inputs=[modelA.input, modelB.input, modelC.input, modelD.input, modelE.input], outputs=output, name="3DCNN")
model.summary()

#todo: evaluate possible alternative optimizers
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt) #alternative: model.compile(loss="mse", optimizer="adam")

# train the model
model.fit(
 	x=[trainA, trainB, trainC, trainD, trainE],
	y=ytrain,
	validation_data=([testA, testB, testC, testD, testE], ytest),
 	epochs=200, batch_size=12, verbose=1)

# make predictions on the testing data
ypred = model.predict([testA, testB, testC, testD, testE])


#plot prediciton
x_ax = range(len(ypred))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

print("Done")


#++++++++++++++NOTES++++++++++++++++++++++++
# #create model as sequential
# inputShape = (60, 40, 7, 1)
#
# # using Sequential
# # model = Sequential()
# # model.add(Conv3D(filters=1, kernel_size=(7, 7, 3), activation="relu", input_shape=inputShape))
# # model.add(Flatten())
# # model.add(Dense(64, activation="relu"))
# # model.add(Dense(1))
#
# #using functional API
# inputs = Input(shape=inputShape)
# x = Conv3D(filters=1, kernel_size=(7, 7, 3), padding="valid", activation="relu", data_format='channels_last')(inputs)
# x = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding="valid")(x)
# x = Conv3D(filters=1, kernel_size=(7, 6, 3), padding="valid", activation="relu")(x)
# x = AveragePooling3D(pool_size=(3, 3, 1), strides=None, padding="valid")(x)
# x = Conv3D(filters=1, kernel_size=(7, 4, 1), padding="valid", activation="relu")(x)
# x = Flatten()(x)
#
# #here combined
# x = Dense(128, activation="relu")(x)
# output = Dense(1, activation="linear")(x)
#
# model = Model(inputs, output)
# model.summary()
#
# model.compile(loss="mse", optimizer="adam")
# model.fit(xtrain, ytrain, batch_size=7, epochs=200, verbose=1)
#
# ypred = model.predict(xtest)
#
# #print("MSE: %.4f" % mean_squared_error(ytest, ypred))
#
# #plot prediciton
# x_ax = range(len(ypred))
# plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
# plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
# plt.legend()
# plt.show()