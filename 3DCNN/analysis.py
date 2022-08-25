import numpy as np
import tensorflow.keras
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import csv

# to be run on msb for real time analysis of videos of ocean surface

# Aufbau:
# die zu analysierenden input arrays liegen in einem Ordner namens "input_analysis", dieser wird aus preprocessing.py gefüllt
# Die ermittelten Charakteristiken (Wellenhöhe und -periode) werden einer csv Datei mit Zeitstempel der aufnahme hinzugefügt

# setup: Model laden

#Ablauf:
#1. Abfragen, ob input daten vorliegen, falls ja einlesen
#2. prediction mit geladenem Modell
#3. prediction der height und period einer csv datei unter dem eigenen zeitstempel hinzufügen
#4. falls erfolgreich input aus dem eingangsordner löschen und einem "trash_data" hinzufügen
#5. falls hinzufügen zu csv datei nicht erfolgreich, dann vielleicht den input in einen "error" ordner legen?

#run on gpu, while testing
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(f"GPU available: " + str(tensorflow.test.is_built_with_cuda()))


# helper functions

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
        # loop over all frames in one cube:
        for k in range(0, input.shape[3]-1):
            prev = input[i, :, :, k, 0]
            next = input[i, :, :, k+1, 0]
            # default params
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            x = flow[:, :, 1]
            result[i, :, :, k, 0] = x
    return result

def rescaling_adjusting_shape(input_array):
    # rescaling data
    image_data = np.array(input_array, dtype="float") / 255.0
    # prepare data and adjust to required shape for convolutions (adding one)
    oldshape = image_data.shape
    newshape = (oldshape[0], oldshape[1], oldshape[2], oldshape[3], 1)
    image_data = image_data.reshape(newshape)
    return image_data

def create_hardwired_input(input_array):
    inputA = gray_scale(input_array)
    inputB = gradient_x(input_array)
    inputC = gradient_y(input_array)
    inputD = optflow_x(input_array)
    inputE = optflow_y(input_array)
    return inputA, inputB, inputC, inputD, inputE

#check whether csv file already exists, if not create a new one with a suitable header
# instatiate .csv file including header if not yet exsisting
def instatiate_csv(filepath_to_csv):
    if not os.path.exists(filepath_to_csv):
        with open(filepath_to_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "wave_height", "wave_period"])

def check_and_load_input(filepath_to_input):
    if os.path.exists(filepath_to_input):
        inputs = os.listdir(filepath_to_input)
    else:
        print("Error: input directory is missing, stopping analysis")

    if len(inputs) < 1:
        print("Error: directory is empty. Need input data for analysis.")
    return inputs

def predict_single_input(input_filename):
    if not input_filename.startswith('.'):
        if input_filename.endswith('.npy'):
            filename = input_filename.split('.')[0]
            filepath = os.path.join(filepath_to_input, input_filename)
            input_data = np.load(filepath)   # input data = nps array shape (112,60,40,7)
            # rescaling and shape adjusting
            image_data = rescaling_adjusting_shape(input_data) #shape (112,60,40,7,1) und Werte zwischen 0 und 1
            inputA, inputB, inputC, inputD, inputE = create_hardwired_input(image_data)   #jeder input (A-E) shape (112,60,40,6/7,1)
            #prediction for eachstack of frame, shape of ypred: list 2, array 112, wave heights: ypred[0] and wave periods: ypred[1]
            ypred = model.predict([inputA, inputB, inputC, inputD, inputE])
            #building the mean of all single predictions for the one recording moment
            mean_height = np.mean(ypred[0])
            mean_period = np.mean(ypred[1])

            predictionname = filename + "_prediction.npy"
            np.save(predictionname, ypred)
            #adding new values to .csv file

            timestamp = create_timestamp_from_filename(filename)
            with open(filepath_to_csv, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, mean_height, mean_period])

            return filename, timestamp, mean_height, mean_period, ypred

def create_timestamp_from_filename(filename):
    day = filename.split('_')[3]
    time = filename.split('_')[4]
    timestamp_string = f"{day}_{time}"
    return timestamp_string

def extract_label(filename):
    # extract label tuple
    last_chars = filename[-9:]
    height, period = last_chars.split("_", 1) # extract wave height and period from name of directory
    height = float(height[0]+height[1]+"."+height[2]+height[3])
    period = float(period[0]+period[1]+"."+period[2]+period[3])
    label_tuple = (height, period)
    return label_tuple

def round_float_list(list, number):
    for i in range(0, len(list)):
        list[i]=round(list[i], number)
        i += 1
    return list

def my_mae(truth, pred_array):
    pred_sum = 0
    diff = 0
    diff_sum = 0
    for i in range(0, len(truth)):
        diff = abs(truth[i] - pred_array[i])
        diff_sum = diff_sum + diff
        pred_sum = pred_sum + pred_array[i]

    return (diff_sum/len(truth)), (pred_sum/len(pred_array))

############# EXECUTION ###########

#config
filepath_to_input = "/home/rueskamp/tmp/pycharm_project_123/input_analysis"
filepath_to_model = "/home/rueskamp/tmp/pycharm_project_123/epochs_100_batch_24_model.h5"
filepath_to_csv = "/home/rueskamp/tmp/pycharm_project_123/predictions.csv"

#for plotting
timestamps = []
mean_heights = []
mean_periods = []
label_heights = []
label_periods = []
yerr_hs = []
yerr_ps = []

#setup
instatiate_csv(filepath_to_csv)
# load model and display architectue
model = load_model(filepath_to_model)
model.summary() # display summary, for checking architecture

inputs = check_and_load_input(filepath_to_input)

# walk through the list inputs
i=0
n=len(inputs)

while i<n:
    filename, timestamp, mean_height, mean_period, single_predicitons = predict_single_input(inputs[i])
    timestamps.append(timestamp)
    mean_heights.append(mean_height)
    mean_periods.append(mean_period)
    yerr_h = (np.std(single_predicitons[0]))/(np.sqrt(len(single_predicitons[0])))
    yerr_h_single = np.std(single_predicitons[0])
    yerr_hs.append(yerr_h_single)
    yerr_p = (np.std(single_predicitons[1]))/(np.sqrt(len(single_predicitons[1])))
    yerr_p_single = np.std(single_predicitons[1])
    yerr_ps.append(yerr_p_single)

    i += 1

    # for evaluation / presentation check with groundtruth
    label_tuple = extract_label(filename)
    label_heights.append(label_tuple[0])
    label_periods.append(label_tuple[1])

    #plotting height:
    #values to plot:
    sigma3h = round(3*round(yerr_h_single, 4), 4)
    uppersigmah = np.full(len(single_predicitons[0]), (round(mean_height, 4)+sigma3h))
    lowersigmah = np.full(len(single_predicitons[0]), (round(mean_height, 4)-sigma3h))

    x1 = np.arange(0, len(single_predicitons[0]), 1)
    label_heights_single = np.full(len(single_predicitons[0]), label_tuple[0])
    mean_heights_single = np.full(len(single_predicitons[0]), mean_height)
    mean_periods_single = np.full(len(single_predicitons[1]), mean_period)

    plt.figure()
    plt.title(f"{timestamp}: Significant Wave Height (Single Measurements)")
    plt.xlabel("Stack of Frames # (7 frames, 10-12 Hz -> 0,6 s per stack)")
    plt.ylabel(f"Significant Wave Height [m]")
    plt.plot(x1, single_predicitons[0], linestyle='None', marker='.', color='blue', label="Single Video Measurement")
    plt.plot(x1, mean_heights_single, '-', color='orange', label="Mean Video Measurement")
    plt.plot(x1, uppersigmah, '--', color='orange', label="3-Sigma Border")
    plt.plot(x1, lowersigmah, '--', color='orange')
    plt.legend(loc="upper right")

    text0 = f"Evaluation of {len(single_predicitons[0])} single measurements, all in [m]: "
    mae = mean_absolute_error(single_predicitons[0], label_heights_single)
    mse = mean_squared_error(single_predicitons[0], label_heights_single)
    text1 = f"Mean Video Measurement: {str(round(mean_height, 4))}; Standard Deviation: {str(round(yerr_h_single, 4))}"
    text2 = f"1-Sigma: {str(round(mean_height, 4))} +- {str(round(yerr_h_single, 4))}; 2-Sigma: {str(round(mean_height, 4))} +- {str(round(2*round(yerr_h_single, 4), 4))}; 3-Sigma: {str(round(mean_height, 4))} +- {str(round(3*round(yerr_h_single, 4), 4))}"
    plt.figtext(0.1, -0.01, text0, horizontalalignment='left', fontsize=10)
    plt.figtext(0.1, -0.06, text1, horizontalalignment='left', fontsize=10)
    plt.figtext(0.1, -0.11, text2, horizontalalignment='left', fontsize=10)
    plt.tight_layout()
    png_name = f"height_{str(i)}.png"
    plt.savefig(png_name, bbox_inches='tight')
    plt.show()

    #plotting period:
    # values to plot:
    sigma3p = round(3 * round(yerr_p_single, 4), 4)
    uppersigmap = np.full(len(single_predicitons[1]), (round(mean_period, 4) + sigma3p))
    lowersigmap = np.full(len(single_predicitons[1]), (round(mean_period, 4) - sigma3p))

    x2 = np.arange(0, len(single_predicitons[1]), 1)
    label_periods_single = np.full(len(single_predicitons[1]), label_tuple[1])
    plt.figure()
    plt.title(f"{timestamp}: Significant Wave Period (Single Measurements)")
    plt.xlabel("Stack of Frames # (7 frames, 10-12 Hz -> 0,6 s per stack)")
    plt.ylabel(f"Significant Wave Period [s]")
    plt.plot(x2, single_predicitons[1], linestyle='None', marker='.', color='green', label="Video Measurement")
    plt.plot(x1, uppersigmap, '--', color='orange', label="3-Sigma Border")
    plt.plot(x1, lowersigmap, '--', color='orange')
    plt.plot(x1, mean_periods_single, '-', color='orange', label="Mean Video Measurement")
    plt.legend(loc="upper right")
    text0 = f"Evaluation of {len(single_predicitons[1])} single measurements, all in [s]: "
    mae = mean_absolute_error(single_predicitons[1], label_periods_single)
    mse = mean_squared_error(single_predicitons[1], label_periods_single)
    text1 = f"Mean Video Measurement: {str(round(mean_period, 4))}; Standard Deviation: {str(round(yerr_p_single, 4))}"
    text2 = f"1-Sigma: {str(round(mean_period, 4))} +- {str(round(yerr_p_single, 4))}; 2-Sigma: {str(round(mean_period, 4))} +- {str(round(2 * round(yerr_p_single, 4), 4))}; 3-Sigma: {str(round(mean_period, 4))} +- {str(round(3 * round(yerr_p_single, 4), 4))}"

    plt.figtext(0.1, -0.01, text0, horizontalalignment='left', fontsize=10)
    plt.figtext(0.1, -0.06, text1, horizontalalignment='left', fontsize=10)
    plt.figtext(0.1, -0.11, text2, horizontalalignment='left', fontsize=10)
    plt.tight_layout()
    png_name = f"period_{str(i)}.png"
    plt.savefig(png_name, bbox_inches='tight')
    plt.show()


# plotting results
x1 = np.arange(0, len(timestamps), 1)
plt.figure()
plt.xlabel("Date of Recording [YYYY-MM-SS_hh-mm-ss]")
plt.ylabel(f"Significant Wave Height [m]")

plt.plot(x1, mean_heights, linestyle='None', marker='o', color = 'blue', label="Video Measurement")
plt.errorbar(x1, mean_heights, yerr_hs, color='blue', ls='none', capsize=3)
plt.plot(x1, label_heights, linestyle='None', marker='x', markersize=7, color='orange', mew=2, label="Buoy Measurement")
plt.xticks(np.arange(len(timestamps)), timestamps, rotation = 30, ha = 'right')
plt.legend(loc="upper left")
text0 = f"Evaluation: "
mae_h = mean_absolute_error(label_heights, mean_heights)
mse_h = mean_squared_error(label_heights, mean_heights)
text1 = f"Mean Absolute Error [m]: {round(mae_h, 4)}; Mean Squared Error [m*m]: {round(mse_h, 4)}"
rounded_list = round_float_list(yerr_hs, 4)
text2 = f"Standard Deviation (sample-wise) [m]:  {str(rounded_list)}"

plt.figtext(0.1, -0.01, text0, horizontalalignment='left', fontsize = 10)
plt.figtext(0.1, -0.06, text1, horizontalalignment='left', fontsize = 10)
plt.figtext(0.1, -0.11, text2, horizontalalignment='left', fontsize = 10)

plt.tight_layout()
plt.savefig('height.png', bbox_inches='tight')
plt.show()

#2.period
plt.figure()
plt.xlabel("Date of Recording [YYYY-MM-SS_hh-mm-ss]")
plt.ylabel(f"Significant Wave Period [s]")
plt.plot(x1, mean_periods, linestyle='None', marker='o', color='green', label="Video Measurement")
plt.errorbar(x1, mean_periods, yerr_ps, color='green', ls='none', capsize=3)
plt.plot(x1, label_periods, marker='x', markersize=7, ls='None', color='orange', mew=2, label="Buoy Measurement")
plt.xticks(np.arange(len(timestamps)), timestamps, rotation = 30, ha = 'right')
plt.legend(loc="upper left")
text0 = f"Evaluation: "
mae = mean_absolute_error(label_periods, mean_periods)
mse = mean_squared_error(label_periods, mean_periods)
text1 = f"Mean Absolute Error [s]: {round(mae, 4)}; Mean Squared Error [s*s]: {round(mse, 4)}"
rounded_list = round_float_list(yerr_ps, 4)
text2 = f"Standard Deviation (sample-wise) [s]:  {str(rounded_list)}"

plt.figtext(0.1, -0.01, text0, horizontalalignment='left', fontsize = 10)
plt.figtext(0.1, -0.06, text1, horizontalalignment='left', fontsize = 10)
plt.figtext(0.1, -0.11, text2, horizontalalignment='left', fontsize = 10)

plt.tight_layout()
plt.savefig('period.png', bbox_inches='tight')
plt.show()


print("Done")
