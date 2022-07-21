import csv
import numpy as np
from pathlib import Path
import os


### help functions ###

# getting the date of the input file (.csv file, downloaded from
# https://seastate.bsh.de/rave/index.jsf?content=download)
def get_date(file):
    with open(file, newline='') as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if reader.line_num == 91:  # first line with date of the data = 91
                date = row[0][0:10]
            else:
                continue
        print("getDate finished")
        return date


# getting the data from the input file
# writing th data into a .txt file with the input date in title
# saving this file into folder 'TXT-Dateien'
def transfer_data(date, file):
    output_dir = os.path.join(os.path.dirname(__file__), 'TXT-Dateien')
    os.makedirs(output_dir, exist_ok=True)
    openfile = open('TXT-Dateien/messdaten-dwr-' + date + '.txt', "w")
    with open(file, newline='') as f:
        reader = csv.reader(f, delimiter=";")
        openfile.write('Date Time VAVH.DWR[m] VAVT.DWR[s]\n')
        for row in reader:
            if reader.line_num >= 91:  # first line with data = 91
                print(row[0], row[1], row[4], file=openfile)  # column 0 = date and time,
                # column 1 = VAVH.DWR data (average height of highest 1/3 waves = H_1/3),
                # column 4 = VAVT.DWR data (average period of highest 1/3 waves = T_1/3)
    openfile.close()
    print("transferData finished")


# reading all .txt files from input directory
# transform the data into numpy-arrays
# save these arrays into folder 'NumPy-Arrays'
def make_array(directory):
    output_dir = os.path.join(os.path.dirname(__file__), 'NumPy-Arrays')
    os.makedirs(output_dir, exist_ok=True)
    directory_name = directory
    open_files = Path(directory_name).glob('*')
    for file in open_files:
        array = np.genfromtxt(file, encoding='ascii', dtype=None, names=True)
        # print(array)
        # print(array.dtype)
        path = os.path.join(output_dir, file.name)
        np.save(path, array)
        print("makeArray finished")


# calling transferData on every file of the input directory
def make_txt(directory):
    directory_name = directory
    open_files = Path(directory_name).glob('*')
    for file in open_files:
        transfer_data(get_date(file), file)
    print("makeTXT finished")


# printing the array (and name, and dimensions)  of the input path
def show_array(arraypath):
    array = np.load(arraypath)
    print(arraypath)
    print(array)
    # print(array.dtype)
    print("showArray finished")


### execution ###

make_txt('D:/Studium Uni Bremen/6. Sem/Systemtechnikprojekt '
         'Seegangmessgerät/bsh/Bojendaten-DWR/nurVAVHundVAVT/pansegan.20220718163715151.csv')
make_array('D:/Studium Uni Bremen/6. Sem/Systemtechnikprojekt Seegangmessgerät/software/TXT-Dateien')
show_array('D:/Studium Uni Bremen/6. Sem/Systemtechnikprojekt '
           'Seegangmessgerät/software/NumPy-Arrays/messdaten-dwr-2022-06-25.txt.npy')
print('execution finished')
