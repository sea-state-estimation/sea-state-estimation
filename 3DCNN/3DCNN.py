import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv3D, Flatten, MaxPooling3D, Dropout, BatchNormalization, AveragePooling3D, GlobalAveragePooling3D, concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from PIL import Image as im
#from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
	#source https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/ (05/2022)
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

def build_output_branch(attribute_name, inputA, inputB, inputC, inputD, inputE):

	modelA = create_branch_model(inputA)
	modelB = create_branch_model(inputB)
	modelC = create_branch_model(inputC)
	modelD = create_branch_model(inputD)
	modelE = create_branch_model(inputE)

	# combine output of the five branches
	combined = concatenate([modelA.output, modelB.output, modelC.output, modelD.output, modelE.output])

	# apply a FC layer and a regression prediction on the combined outputs
	z = Dense(128, activation="relu")(combined)
	output = Dense(1, activation="linear", name=attribute_name)(z)

	# our model will accept the inputs of the 5 branches and then output a single value
	model = Model(inputs=[modelA.input, modelB.input, modelC.input, modelD.input, modelE.input], outputs=output, name=attribute_name)

	return model

#additionaly to visualize the extracted image sections, there are 4 section per original frame, time series of 7 frames
def save_time_series_section_as_image(position):
	n = position # 4 entries belong to one original frame

	image0 = im.fromarray(image_data_raw[n, :, :, 0])
	image1 = im.fromarray(image_data_raw[n, :, :, 1])
	image2 = im.fromarray(image_data_raw[n, :, :, 2])
	image3 = im.fromarray(image_data_raw[n, :, :, 3])
	image4 = im.fromarray(image_data_raw[n, :, :, 4])
	image5 = im.fromarray(image_data_raw[n, :, :, 5])
	image6 = im.fromarray(image_data_raw[n, :, :, 6])

	# saving the final output as a PNG file
	image0.save('image_section_0.png')
	image1.save('image_section_1.png')
	image2.save('image_section_2.png')
	image3.save('image_section_3.png')
	image4.save('image_section_4.png')
	image5.save('image_section_5.png')
	image6.save('image_section_6.png')

####################EXECUTION#######################

#run on gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(f"GPU available: " + str(tensorflow.test.is_built_with_cuda()))

# load data, adjust path to .npy array, required shape (n, 60, 40, 7)
image_data_raw = np.load("/home/rueskamp/tmp/pycharm_project_123/dataset_guarares.npy")
# rescale data to interval [0, 1], can be checked with max_raw = np.amax(image_data_raw)
image_data = np.array(image_data_raw, dtype="float") / 255.0

# additionally for saving images as a time series cube as extracted from original image, e.g. at time series number 3, uncomment next line if desired
# save_time_series_section_as_image(3)

labels = np.load("/home/rueskamp/tmp/pycharm_project_123/labels_guarares.npy")

# prepare data and adjust to required shape for convolutions (adding one dimension)
oldshape = image_data.shape
newshape = (oldshape[0], oldshape[1], oldshape[2], oldshape[3], 1)
image_data = image_data.reshape(newshape)

# splitting data into training and test data (85% vs 15%)
# inputdata x (n, 60, 40, 7, 1), labels y (2-dim) therefore resulting shape (n, 2), height ytrain[:,0], period ytrain[:,1]
xtrain, xtest, ytrain, ytest = train_test_split(image_data, labels, test_size=0.15)

#creating multiple inputs hardwired as suggested in Ji 2010 (grayscale, gradientx, gradienty, optflowx optflowy)
trainA, trainB, trainC, trainD, trainE, testA, testB, testC, testD, testE = create_hardwired_input(xtrain, xtest)


######BUILDING MODEL##############

# 2 different inputShapes as optflow will reduce depth by one to 6, others (grayscale, gradientx/y) keep 7
inputShape1 = (60, 40, 7, 1)  # grayscale and gradient x, y
inputShape2 = (60, 40, 6, 1)  # optflow

# Inputs for each hardwired version (A-E)
inputA = Input(shape=inputShape1)
inputB = Input(shape=inputShape1)
inputC = Input(shape=inputShape1)
inputD = Input(shape=inputShape2)
inputE = Input(shape=inputShape2)

heightBranch_model = build_output_branch("height", inputA, inputB, inputC, inputD, inputE)
periodBranch_model = build_output_branch("period", inputA, inputB, inputC, inputD, inputE)

model = Model(inputs=[inputA, inputB, inputC, inputD, inputE], outputs=[heightBranch_model.output, periodBranch_model.output], name="3DCNN")
model.summary()

opt = Adam(lr=1e-3, decay=1e-3 / 200)

# not used yet, difficult to manage with multiple inputs
# aug = ImageDataGenerator(
#   rotation_range=20,
#   zoom_range=0.15,
#   width_shift_range=0.2,
#   height_shift_range=0.2,
#   shear_range=0.15,
#   horizontal_flip=True,
#   fill_mode="nearest")

loss_function = "mae"
model.compile(loss=loss_function, metrics=['mean_absolute_error', 'mean_squared_error'], optimizer=opt) #alternative: model.compile(loss="mse", optimizer="adam")

# define hyperparameter, best option for us batchsize = 24, epochs = 100
EPOCHS = 100
BATCHSIZE = 24

# unsuccessful trying to include data augmentation, failing
# model.fit(
# 	x=aug.flow(trainX, trainY, batch_size=BS),
# 	validation_data=(testX, testY),
# 	steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS)

# #test for augmentation
# H = model.fit(
#  	x=aug.flow(x=[trainA, trainB, trainC, trainD, trainE], y=[ytrain[:,0], ytrain[:,1]], batch_size=12),
# 	validation_data=([testA, testB, testC, testD, testE], [ytest[:,0], ytest[:,1]]),
#  	epochs=EPOCHS, verbose=1)

# train the model
H = model.fit(
 	x=[trainA, trainB, trainC, trainD, trainE],
	y=[ytrain[:,0], ytrain[:,1]],
	validation_data=([testA, testB, testC, testD, testE], [ytest[:,0], ytest[:,1]]),
 	epochs=EPOCHS, batch_size=BATCHSIZE, verbose=1)

# create name for saving model, history, prediction
name = f"epochs_{EPOCHS}_batch_{BATCHSIZE}_"
modelname = name+"model.h5"
historyname = name + "history.npy"

#to load use: model = load_model("/home/rueskamp/tmp/pycharm_project_123/threeD.h5")
model.save(modelname)
#to load use: history = np.load('/home/rueskamp/tmp/pycharm_project_123/my_history.npy',allow_pickle='TRUE').item()
np.save(historyname, H.history)

# plot the training evaluation (losses train and test)
N = np.arange(0, EPOCHS)
plt.figure()
plt.plot(N, H.history["height_loss"], label="height loss [m]")
plt.plot(N, H.history["val_height_loss"], label="height loss val [m]")
plt.plot(N, H.history["val_period_loss"], label="period loss val [s]")
plt.plot(N, H.history["period_loss"], label="period loss [s]")
plt.title("Training Evaluation")
plt.xlabel("Epoch #")
plt.ylabel(f"Loss ({loss_function})")
plt.legend(loc="upper right")
plt.savefig(f"plot_training_evaluation_{EPOCHS}_{BATCHSIZE}")
plt.show()
plt.clf()

print("Done")