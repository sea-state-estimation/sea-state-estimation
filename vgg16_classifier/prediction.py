import numpy as np
import tensorflow.keras
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#run on gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(f"GPU available: " + str(tensorflow.test.is_built_with_cuda()))

# remote working directory:
current_directory = "/home/rueskamp/project"
# used dataset
dataset_name = "dataset_waves_Guimaraes2020"

# path to dataset starting in working directory
dataset_path = os.path.join(current_directory, f"datasets/"+dataset_name)
dataset_path_test = os.path.join(dataset_path, "test")

#tsdata = ImageDataGenerator()
#testdata = tsdata.flow_from_directory(directory=dataset_path_test, target_size=(224,224))

# load model
model = load_model("/home/rueskamp/project/src/vgg16_1.h5")
model.summary()

#prediction = model.predict(testdata)

dir = os.listdir(dataset_path_test)

# loop through all images in the selected folder
for img_src in dir:
    if not img_src.startswith('.'):
        img = image.load_img(
            "/Users/rueskamp/Documents/Studium SE/05_WS21/Projekt_See/codebase/vgg16_catdog/src/01_425.jpg",
            target_size=(224, 224))
        img = np.asarray(img)
        plt.imshow(img)
        plt.show()

        img = np.expand_dims(img, axis=0)

        saved_model = load_model("vgg16_1.h5")
        output = saved_model.predict(img)
        # print(f"output: "+ output)
        print(f"probability for label 0: " + str(output[0][0]))
        print(f"probability for label 1: " + str(output[0][1]))
        print(f"probability for label 2: " + str(output[0][2]))
print("Done")

