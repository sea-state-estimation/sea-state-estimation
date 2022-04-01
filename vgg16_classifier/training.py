import tensorflow.keras
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.models import load_model

#from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, accuracy_score, precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import plot_confusion_matrix

#run on gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(f"GPU available: " + str(tensorflow.test.is_built_with_cuda()))

#Anleitung: https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

# remote working directory:
current_directory = "/home/rueskamp/project"

# used dataset
dataset_name = "dataset_waves_Guimaraes2020"

# path to dataset starting in working directory
dataset_path = os.path.join(current_directory, f"datasets/"+dataset_name)
dataset_path_train = os.path.join(dataset_path, "train")
dataset_path_valid = os.path.join(dataset_path, "valid")

trdata = ImageDataGenerator()
tsdata = ImageDataGenerator()

traindata = trdata.flow_from_directory(directory=dataset_path_train,target_size=(224,224), batch_size=64)
testdata = tsdata.flow_from_directory(directory=dataset_path_valid, target_size=(224,224), batch_size=1)



model = Sequential()

model.add(Conv2D(input_shape=(224,224,3), filters=64, kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=3, activation="softmax"))

metric = 'acc'

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=tensorflow.keras.losses.categorical_crossentropy, metrics=[metric])

model.summary()

num_epochs = 100
num_steps_per_epoch = 10
num_validation_steps = 10
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor=[metric], verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
early = EarlyStopping(monitor=metric, min_delta=0, patience=20, verbose=1, mode='auto')

#hist = model.fit_generator(steps_per_epoch=100,x=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])
hist = model.fit(x=traindata, epochs=num_epochs, verbose=1, validation_data=testdata, shuffle=False, initial_epoch=0, steps_per_epoch=num_steps_per_epoch, validation_steps=num_validation_steps)
#hist = model.fit(x=traindata, epochs=num_epochs, verbose=1, callbacks=[checkpoint,early], validation_data=testdata, shuffle=False, initial_epoch=0, steps_per_epoch=num_steps_per_epoch, validation_steps=num_validation_steps)
#hist = model.fit(x=traindata, epochs=5, verbose=1, callbacks=[checkpoint,early], validation_data=testdata, shuffle=False, initial_epoch=0, steps_per_epoch=2, validation_steps=10)

#(x=traindata, epochs=100, verbose=1, callbacks=[checkpoint,early], validation_data=testdata, shuffle=True, initial_epoch=0, steps_per_epoch=100)

#displaying accuracy and loss each training and validation data
plt.figure(figsize=(15,5))
plt.plot(hist.history[metric])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0,3)
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])

#adding training details to graphic
text0 = f"Name of dataset: " + dataset_name
text1 = f"Number of epochs: " + str(num_epochs)
text2 = f"Steps per epoch: " + str(num_steps_per_epoch)
text3 = f"Number of validation steps: " + str(num_validation_steps)
plt.figtext(0.1, -0.01, text0, horizontalalignment='left')
plt.figtext(0.1, -0.06, text1, horizontalalignment='left')
plt.figtext(0.1, -0.11, text2, horizontalalignment='left')
plt.figtext(0.1, -0.16, text3, horizontalalignment='left')

#save graphic as .png
filename = f""+str(num_epochs)+"_"+str(num_steps_per_epoch)+"_vx.png"
directorypath = "/home/rueskamp/project/"
filepath = os.path.join(directorypath, filename)

plt.savefig(filepath, bbox_inches='tight')
plt.show()

###############PREDICTION 1###########
prediction = []
groundtruth = []

#enter name of testset (test or test_mixed)
test_name = "test"
dataset_path_test = os.path.join(dataset_path, test_name)

dir = os.listdir(dataset_path_test)

# loop through all images in the selected folder
for img_src in dir:
    if not img_src.startswith('.'):
        # tatsächliches Label zu grounthruth hinzufügen
        label = img_src[:2]
        if label == '00':
            groundtruth.append(0)
        elif label == '01':
            groundtruth.append(1)
        elif label == '02':
            groundtruth.append(2)
        else:
            print('Testbild war nicht gelabelt!')

        #Bild laden und als array dem model übergeben
        img_path = os.path.join(dataset_path_test, img_src)
        img = image.load_img( img_path, target_size=(224, 224))
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        #saved_model = load_model("vgg16_1.h5")
        output = model.predict(img)

        # aus der prediction die wahrscheinlichste Klasse extrahieren
        if output[0][0] >= output[0][1]:
            if output[0][0] >= output[0][2]:
                prediction.append(0)
            else:
                prediction.append(2)
        else:
            if output[0][1] >= output[0][2]:
                prediction.append(1)
            else:
                prediction.append(2)

#predictions and groundtruth extraction done

#evaluation metrics overall accuracy, precision, recall
acc = accuracy_score(groundtruth, prediction)
precision_macro = precision_score(groundtruth, prediction, average='macro')
recall_macro = recall_score(groundtruth, prediction, average='macro')

#confusion matrices
cm = confusion_matrix(groundtruth, prediction)
cm_norm = confusion_matrix(groundtruth, prediction, normalize='true')
#displaying cms
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)

#evaluation metrics per class
tp0 = cm[0][0]
fp0 = cm[1][0] + cm[2][0] #übereinander stehend, im np arra jeweils der erste eintrag
fn0 = cm[0][1] + cm[0][2] #horizontal nebeneinander
tn0 = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
accuracy0 = (tp0+tn0)/(tp0+fp0+fn0+tn0)
precision0 = tp0/(tp0+fp0)
recall0 = tp0/(tp0+fn0)

tp1 = cm[1][1]
fp1 = cm[0][1] + cm[1][2]
fn1 = cm[1][0] + cm[1][2]
tn1 = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
accuracy1 = (tp1+tn1)/(tp1+fp1+fn1+tn1)
precision1 = tp1/(tp1+fp1)
recall1 = tp1/(tp1+fn1)

tp2 = cm[2][2]
fp2 = cm[0][2] + cm[1][2]
fn2 = cm[2][0] + cm[2][1]
tn2 = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
accuracy2 = (tp2+tn2)/(tp2+fp2+fn2+tn2)
precision2 = tp2/(tp2+fp2)
recall2 = tp2/(tp2+fn2)

# put first graphic together
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
text0 = f"Testset: "+test_name
text1 = f"Class 0: accuracy: " + str(round(accuracy0,2)) + " / precision: " + str(round(precision0, 2)) + " / recall: " + str(round(recall0, 2)) + " / tp: " + str(tp0) + ", tn: " + str(tn0) + ", fp: " + str(fp0) + ", fn: "+ str(fn0)
text2 = f"Class 1: accuracy: " + str(round(accuracy1,2)) + " / precision: " + str(round(precision1, 2)) + " / recall: " + str(round(recall1, 2)) + " / tp: " + str(tp1) + ", tn: " + str(tn1) + ", fp: " + str(fp1) + ", fn: "+ str(fn1)
text3 = f"Class 2: accuracy: " + str(round(accuracy2,2)) + " / precision: " + str(round(precision2, 2)) + " / recall: " + str(round(recall2, 2)) + " / tp: " + str(tp2) + ", tn: " + str(tn2) + ", fp: " + str(fp2) + ", fn: "+ str(fn2)
plt.figtext(0.1, -0.01, text0, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.06, text1, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.11, text2, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.16, text3,  horizontalalignment='left', fontsize = 8)
#saving as .png
filename = "Evaluation.png"
filepath = os.path.join(directorypath, filename)
plt.savefig(filepath, bbox_inches='tight')
plt.show()

#put second graphic together
disp_norm.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix normalized')
text0 = f"Testset: "+test_name
text1 = f"Overall-accuracy: " + str(round(acc, 2))
text2 = f"Overall-precision: " + str(round(precision_macro, 2))
text3 = f"Overall-recall: " + str(round(precision_macro, 2))
plt.figtext(0.1, -0.01, text0, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.06, text1, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.11, text2, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.16, text3, horizontalalignment='left', fontsize = 8)
#saving as .png
filename = "Evaluation_norm.png"
filepath = os.path.join(directorypath, filename)
plt.savefig(filepath, bbox_inches='tight')
plt.show()

print("Done 1")


###############PREDICTION 2###########
prediction = []
groundtruth = []

#enter name of testset (test or test_mixed)
test_name = "test_mixed"
dataset_path_test = os.path.join(dataset_path, test_name)

dir = os.listdir(dataset_path_test)

# loop through all images in the selected folder
for img_src in dir:
    if not img_src.startswith('.'):
        # tatsächliches Label zu grounthruth hinzufügen
        label = img_src[:2]
        if label == '00':
            groundtruth.append(0)
        elif label == '01':
            groundtruth.append(1)
        elif label == '02':
            groundtruth.append(2)
        else:
            print('Testbild war nicht gelabelt!')

        #Bild laden und als array dem model übergeben
        img_path = os.path.join(dataset_path_test, img_src)
        img = image.load_img( img_path, target_size=(224, 224))
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        #saved_model = load_model("vgg16_1.h5")
        output = model.predict(img)

        # aus der prediction die wahrscheinlichste Klasse extrahieren
        if output[0][0] >= output[0][1]:
            if output[0][0] >= output[0][2]:
                prediction.append(0)
            else:
                prediction.append(2)
        else:
            if output[0][1] >= output[0][2]:
                prediction.append(1)
            else:
                prediction.append(2)

#predictions and groundtruth extraction done

#evaluation metrics overall accuracy, precision, recall
acc = accuracy_score(groundtruth, prediction)
precision_macro = precision_score(groundtruth, prediction, average='macro')
recall_macro = recall_score(groundtruth, prediction, average='macro')

#confusion matrices
cm = confusion_matrix(groundtruth, prediction)
cm_norm = confusion_matrix(groundtruth, prediction, normalize='true')
#displaying cms
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)

#evaluation metrics per class
tp0 = cm[0][0]
fp0 = cm[1][0] + cm[2][0] #übereinander stehend, im np arra jeweils der erste eintrag
fn0 = cm[0][1] + cm[0][2] #horizontal nebeneinander
tn0 = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
accuracy0 = (tp0+tn0)/(tp0+fp0+fn0+tn0)
precision0 = tp0/(tp0+fp0)
recall0 = tp0/(tp0+fn0)

tp1 = cm[1][1]
fp1 = cm[0][1] + cm[1][2]
fn1 = cm[1][0] + cm[1][2]
tn1 = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
accuracy1 = (tp1+tn1)/(tp1+fp1+fn1+tn1)
precision1 = tp1/(tp1+fp1)
recall1 = tp1/(tp1+fn1)

tp2 = cm[2][2]
fp2 = cm[0][2] + cm[1][2]
fn2 = cm[2][0] + cm[2][1]
tn2 = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
accuracy2 = (tp2+tn2)/(tp2+fp2+fn2+tn2)
precision2 = tp2/(tp2+fp2)
recall2 = tp2/(tp2+fn2)

# put first graphic together
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
text0 = f"Testset: "+test_name
text1 = f"Class 0: accuracy: " + str(round(accuracy0,2)) + " / precision: " + str(round(precision0, 2)) + " / recall: " + str(round(recall0, 2)) + " / tp: " + str(tp0) + ", tn: " + str(tn0) + ", fp: " + str(fp0) + ", fn: "+ str(fn0)
text2 = f"Class 1: accuracy: " + str(round(accuracy1,2)) + " / precision: " + str(round(precision1, 2)) + " / recall: " + str(round(recall1, 2)) + " / tp: " + str(tp1) + ", tn: " + str(tn1) + ", fp: " + str(fp1) + ", fn: "+ str(fn1)
text3 = f"Class 2: accuracy: " + str(round(accuracy2,2)) + " / precision: " + str(round(precision2, 2)) + " / recall: " + str(round(recall2, 2)) + " / tp: " + str(tp2) + ", tn: " + str(tn2) + ", fp: " + str(fp2) + ", fn: "+ str(fn2)
plt.figtext(0.1, -0.01, text0, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.06, text1, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.11, text2, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.16, text3,  horizontalalignment='left', fontsize = 8)
#saving as .png
filename = "Evaluation_mixed.png"
filepath = os.path.join(directorypath, filename)
plt.savefig(filepath, bbox_inches='tight')
plt.show()

#put second graphic together
disp_norm.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix normalized')
text0 = f"Testset: "+test_name
text1 = f"Overall-accuracy: " + str(round(acc, 2))
text2 = f"Overall-precision: " + str(round(precision_macro, 2))
text3 = f"Overall-recall: " + str(round(precision_macro, 2))
plt.figtext(0.1, -0.01, text0, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.06, text1, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.11, text2, horizontalalignment='left', fontsize = 8)
plt.figtext(0.1, -0.16, text3, horizontalalignment='left', fontsize = 8)
#saving as .png
filename = "Evaluation_mixed_norm.png"
filepath = os.path.join(directorypath, filename)
plt.savefig(filepath, bbox_inches='tight')
plt.show()

print("Done 2")