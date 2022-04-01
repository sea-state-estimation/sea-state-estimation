import cv2
import os

#configure:
#Name of directory where images before cropping are
directory_name = "AA_2015_2015-03-05_10-35-00_12Hz_jpg"

#create a destination_folder if not yet existing
current_directory = os.getcwd()
directory_src = os.path.join(current_directory, directory_name)

directory_name_new = directory_name + "_cropped"
final_directory = os.path.join(current_directory, directory_name_new)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

dir = os.listdir(directory_src)

for img_src in dir:
    if not img_src.startswith('.'):
        #name, ending = img_src.split(".")
        path = os.path.join(directory_src, img_src)
        img = cv2.imread(path) #read in image
        img = img[220: 1858, 270: 2186] #crop image
        #filename_new = name + ".jpg"
        os.chdir(final_directory)
        cv2.imwrite(img_src, img)

print("Done")