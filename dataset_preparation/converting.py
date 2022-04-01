# import Image from wand.image module
from wand.image import Image
from wand.display import display
import os

directory_src = 'AA_2015_2015-03-05_10-35-00_12Hz'

#create a destination_folder if not yet existing
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, "AA_2015_2015-03-05_10-35-00_12Hz_jpg")
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

dir = os.listdir(directory_src)

for img_src in dir:
    if not img_src.startswith('.'):
        name, ending = img_src.split(".")

# Read .png image using Image() function
    #with Image(filename =name + ".tif") as img:
        with Image(filename =os.path.join(directory_src, img_src)) as img:
            img.format = 'jpeg'
            filename_new = name + ".jpg"
            img.save(filename=os.path.join(final_directory, filename_new))

print("Done")