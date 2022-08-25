import cv2
import os
import numpy as np

#Helpers functions

# turn single video into gray-scale numpy array, param: full path to video-file
# shape (z, y, x) aka (depth, height, width)

# shape of returned array: (number of frame, height, width), no channels like rgb as already gray values
def video_to_ndarray_gray(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray is (H, W)
            frames.append(gray)
    video_array = np.stack(frames, axis=0)  # dimensions (N, H, W)
    return video_array


# crop/extract array into desired size (for 3DCNN shape (60,40,7)) values from paper Ji 2013
# params: video array to be used, starting coordinates (z_start, y_start, x_start) and width of desired extract (z_num, y_num, x_num)
def extract_input_cube(video_array, z_start, y_start, x_start, desired_depth, desired_height, desired_width):
    input_cube = video_array[z_start:(z_start+desired_depth), y_start:(y_start+desired_height), x_start:(x_start+desired_width)]
    input_cube = np.swapaxes(input_cube, 0, 2) # (40,60,7)
    input_cube = np.swapaxes(input_cube, 0, 1) # (60,40,7)
    return input_cube


# shape of video_array: (depth/number of frames, height, width)
# create multiple cubes from video-array, as format is much smaller than original shape
# getting sections around the image center, to be done, right now some error, as 2 sections are the same, outdated as create_four_cubes() is used instead
def create_multiple_cubes(video_array, num_per_frame_h, num_per_frame_w, num_timeseries, desired_depth, desired_height, desired_width):

    height = video_array.shape[1]
    width = video_array.shape[2]
    mp = (height/2, width/2)

    stacked_cubes = []
    number = num_per_frame_h * num_per_frame_w

    if number>0:
        #process different timeseries
        if video_array.shape[0] <= 7:
            num_timeseries = 1

        for j in range(0, num_timeseries):
            steps_z = int(video_array.shape[0] / num_timeseries)
            start_z = j * steps_z

            #iterate height
            for i in range(0, num_per_frame_h):
                steps_h = int(video_array.shape[1] / num_per_frame_h)
                start_h = i * steps_h
                #iterate width
                for k in range(0, num_per_frame_w):
                    steps_w = int(video_array.shape[2] / num_per_frame_w)
                    start_w = i*steps_w
                    stacked_cubes.append(extract_input_cube(video_array, start_z, start_h, start_w, desired_depth, desired_height, desired_width)) # list of arrays (60,40,7)
    return stacked_cubes


def create_four_cubes(video_array, num_timeseries, desired_depth, desired_height, desired_width):
    height = video_array.shape[1]
    width = video_array.shape[2]
    mp = (height / 2, width / 2)

    stacked_cubes = []
    number = 4

    # process different timeseries
    if video_array.shape[0] <= 7:
        num_timeseries = 1

    for j in range(0, num_timeseries):
        steps_z = int(video_array.shape[0] / num_timeseries)
        start_z = j * steps_z

        #1 top left:
        start_h1 = int(mp[1] - 25 - desired_height)
        start_w1 = int(mp[0] - 50 - desired_width)

        # 2 top right:
        start_h2 = int(mp[1] - 25 - desired_height)
        start_w2 = int(mp[0] + 50)

        # 3 bottom left:
        start_h3 = int(mp[1] + 25)
        start_w3 = int(mp[0] - 50 - desired_width)

        # 4 bottom right:
        start_h4 = int(mp[1] + 25)
        start_w4 = int(mp[0] + 50)

        stacked_cubes.append(extract_input_cube(video_array, start_z, start_h1, start_w1, desired_depth, desired_height, desired_width))
        stacked_cubes.append(extract_input_cube(video_array, start_z, start_h2, start_w2, desired_depth, desired_height, desired_width))
        stacked_cubes.append(extract_input_cube(video_array, start_z, start_h3, start_w3, desired_depth, desired_height, desired_width))
        stacked_cubes.append(extract_input_cube(video_array, start_z, start_h4, start_w4, desired_depth, desired_height, desired_width))
    return stacked_cubes


def save_array(array, filename):
    np.save(filename, array)


# stack arrays from multiple different videos
def prepare_dataset(filepath_to_videos, num_per_frame_h, num_per_frame_w, num_timeseries, desired_depth, desired_height, desired_width):
    dataset = []
    # loop through different videos from given directory
    for file in os.listdir(filepath_to_videos):
        array_gray = video_to_ndarray_gray(os.path.join(filepath_to_videos, file))

        # extract desiered number of cubes, e. g. 4 per frame, each 3 timeseries
        cubes_from_one_vid = create_multiple_cubes(array_gray, num_per_frame_h, num_per_frame_w, num_timeseries, desired_depth, desired_height, desired_width)
        dataset = dataset+cubes_from_one_vid # list of n cube each (60,40,7)
    return dataset

# function to build a single input array from extern dataset and creating a label-matrix at the same time
def prepare_dataset_extern(filepath_to_videos, num_per_frame_h, num_per_frame_w, num_timeseries, desired_depth, desired_height, desired_width):
    dataset = []
    labels = []

    # loop through different videos from given directory
    # start uncomment
    # for directory in os.listdir(filepath_to_videos):
    #     print(f"next directory: {directory}")
    #     k = 0
    #     last_chars = directory[-9:]
    #     height, period = last_chars.split("_", 1) # extract wave height and period from name of directory
    #     height = float(height[0]+height[1]+"."+height[2]+height[3])
    #     period = float(period[0]+period[1]+"."+period[2]+period[3])
    #     tuple = (height, period)
    #
    #     # add a label tuple to "labels" for each file that is in
    #     #list = os.listdir(os.path.join(filepath_to_videos, directory))  # dir is your directory path
    #     #number_files = len(list)
    #     #for i in range (0, number_files):
    #     #    labels.append(tuple)
    #
    #     #my_result = [(tuple,) * number_files]
    #     #[(height, period), (height, period)]
    #     #labels = labels+(height, period)
    #     list = os.listdir(os.path.join(filepath_to_videos, directory))
    #     number_files = len(list)
    #     number_loops = int(number_files/7)
    #
    #     for l in range(0, number_loops):
    #         cube = []
    #         for j in range (k, k+7):
    #             img = cv2.imread(os.path.join(filepath_to_videos, os.path.join(directory, list[j])))
    #             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #             cube.append(gray)
    #         video_array = np.stack(cube, axis=0)
    #
    #         cubes_from_one_vid = create_four_cubes(video_array, num_timeseries, desired_depth, desired_height, desired_width)
    #         dataset = dataset + cubes_from_one_vid  # list of n cube each (60,40,7)
    #         #dataset.append(video_array)
    #         k += 7
    #         number = len(cubes_from_one_vid)
    #         for p in range(0, number):
    #             labels.append(tuple)
# end uncomment


    # start uncomment if you give path directly to videos to be converted to a .npy array (in total insert name of directory 4 time, 1x here, 3x below)
    directory = "BS_2013_2013-09-30_10-20-01_12Hz_jpg_0065_0570"
    k = 0
    last_chars = directory[-9:]
    height, period = last_chars.split("_", 1)  # extract wave height and period from name of directory
    height = float(height[0] + height[1] + "." + height[2] + height[3])
    period = float(period[0] + period[1] + "." + period[2] + period[3])
    tuple = (height, period)

    list = os.listdir(filepath_to_videos)
    number_files = len(list)
    number_loops = int(number_files / 7)

    for l in range(0, number_loops):
        cube = []
        for j in range(k, k + 7):
            img = cv2.imread(os.path.join(filepath_to_videos, list[j]))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cube.append(gray)
        video_array = np.stack(cube, axis=0)

        cubes_from_one_vid = create_four_cubes(video_array, num_timeseries, desired_depth, desired_height,
                                               desired_width)
        dataset = dataset + cubes_from_one_vid  # list of n cube each (60,40,7)

        k += 7
        number = len(cubes_from_one_vid)
        for p in range(0, number):
            labels.append(tuple)
    # end uncomment

    return dataset, labels


######### EXECUTION #################
#vid_array = video_to_ndarray_gray("/home/rueskamp/tmp/pycharm_project_123/videos/2022-06-01_11-11-50+02-00.h264")
#input = extract_input_cube(vid_array, 10, 10, 10, 7, 60, 40)

#save_array(input, "/home/rueskamp/tmp/pycharm_project_123/test_array")
#loaded = np.load("/home/rueskamp/tmp/pycharm_project_123/test_array.npy")

#dataset = prepare_dataset("/home/rueskamp/tmp/pycharm_project_123/videos", 2, 2, 3, 7, 60, 40)

# uncomment section before and next command if one npy array is supposed to be created, browsing trough a whole directory
#dataset, labels = prepare_dataset_extern("/home/rueskamp/tmp/pycharm_project_123/Dataset_Guarares/train", 2, 2, 3, 7, 60, 40)
dataset, labels = prepare_dataset_extern("/home/rueskamp/tmp/pycharm_project_123/Dataset_Guarares/test/BS_2013_2013-09-30_10-20-01_12Hz_jpg_0065_0570", 2, 2, 3, 7, 60, 40)
save_array(dataset,"/home/rueskamp/tmp/pycharm_project_123/test_BS_2013_2013-09-30_10-20-01_12Hz_jpg_0065_0570")

#fake_labels_test = np.concatenate([([i]*12) for i in [0.1,0.8,1.5]], axis=0)
save_array(labels,"/home/rueskamp/tmp/pycharm_project_123/testlabels_BS_2013_2013-09-30_10-20-01_12Hz_jpg_0065_0570")

print('Done')