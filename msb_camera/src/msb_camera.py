import datetime as dt
import pytz
import zmq
import sys
import pickle
import os
import logging
from pathlib import Path
import time


# Vorlage für Sockets und Datenempfang (Code von Andreas): https://github.com/ahaselsteiner/msb-mqtt/blob/cb695298f4c4df65e5df45c128beecb90e682457/src/msb_mqtt.py
# Vorlage für isNowInTimePeriod (StackOverflow, 19.05.2022): https://stackoverflow.com/questions/10048249/how-do-i-determine-if-current-time-is-within-a-specified-range-using-pythons-da


#### STATIC VARIABLES #####

BASE_VIDEO_PATH_1 = '/media/usbstick1/'
BASE_VIDEO_PATH_2 = '/media/usbstick2/'
# zero mq connection infos
ipc_protocol = 'tcp://127.0.0.1'
ipc_port = '5556'

default_video_height = 1920
default_video_width = 1080
default_video_framerate = 12
default_pause = 1200 # in s, 20 min = 1200 s

# 5 s = 5000 ms, # 1 minute = 6000, 3 min = 180000, 5 min = 300000, 15 min = 900000 ms
default_video_duration = 600000 # in ms, 10 min = 600 000 ms

#### FUNCTIONS #####

def is_now_in_time_period(start_time, end_time, now_time):
    if start_time < end_time:
        return now_time >= start_time and now_time <= end_time
    else:
        #Over midnight:
        return now_time >= start_time or now_time <= end_time

def receiving_data(ipc_protocol, ipc_port):
    connect_to = f'{ipc_protocol}:{ipc_port}'
    ctx = zmq.Context()
    zmq_socket = ctx.socket(zmq.SUB)

    try:
        zmq_socket.connect(connect_to)
    except Exception as e:
        logging.fatal(f'Failed to bind to zeromq socket: {e}')
        sys.exit(-1)

    # Subscribe to all available data.
    zmq_socket.setsockopt(zmq.SUBSCRIBE, b'')
    # receiving data
    (zmq_topic, data) = zmq_socket.recv_multipart()
    zmq_topic = zmq_topic.decode('utf-8')
    data = pickle.loads(data)
    return zmq_topic, data

def set_sytem_date(datetime_object):
    """Update System time and date with received gps signal dealing with missing internet connection
    """
    #date_string = f'{datetime_object.year}-{datetime_object.month}-{datetime_object.day} {datetime_object.hour}:{datetime_object.minute}:{datetime_object.second}'
    date_string = f'"{datetime_object.date()} {datetime_object.time()}"'
    os.system(f'sudo date -s {date_string}')
    logging.debug(f'System date was set to: {date_string}')

def recording(height, width, framerate, duration, filename):
    # check whether usbstick 1 is connected, therefore BASE_VIDEO_PATH_1 exist
    # requirement: each usb sticks containing an empty .txt file calles usbstick{number}_connected.txt
    if os.path.isfile(BASE_VIDEO_PATH_1+'usbstick1_connected.txt'):
        path_in_use = BASE_VIDEO_PATH_1
    else:
        if os.path.isfile(BASE_VIDEO_PATH_2+'usbstick2_connected.txt'):
            path_in_use = BASE_VIDEO_PATH_2
        else:
            # if both usbsticks are not recognised try rebooting
            logging.info('System did not find a memory stick. Rebooting now.')
            os.system(f'sudo shutdown -r')

    file_path_name = path_in_use + filename
    logging.info(f'Recording path is: {file_path_name}. Recording should start now.')

    recording_cli_cmd = f'sudo libcamera-vid -n -t {duration} --flush --height {height} --width {width} --framerate {framerate} -o {file_path_name}.h264 --save-pts {file_path_name}_timestamps.txt'
    # execute the recording_cli_cmd as system shell cmd
    os.system(recording_cli_cmd)
    logging.info(f'Recording stopped.')

    logging.info(f'Going to pause for {default_pause} seconds')
    time.sleep(default_pause)
    logging.info('Pausing stopped')

def check_and_create_logname():
    log_filename = 'camera-script.log'
    my_file = Path(log_filename)
    # if a log file already exist, find a new filename
    if my_file.is_file():
        for i in range(0, 900):
            my_file_2 = Path(f'camera-script-{i}.log')
            if my_file_2.is_file() == False : #filename does not yet exist, therefore use it
                log_filename = f'camera-script-{i}.log'
                return log_filename
    return log_filename

def main():
    log_filename = check_and_create_logname()
    logging.basicConfig(format='%(asctime)s %(message)s',
                        filename='/home/pi/motion-sensor-box/src/camera/src/'+log_filename,
                        encoding='utf-8',
                        level=logging.DEBUG)
    logging.info(f'Camera module has started')
    logging.info(f'Trying to bind zmq to {ipc_protocol}:{ipc_port}')
    try:
        while True:
            zmq_topic, data = receiving_data(ipc_protocol, ipc_port)

            if zmq_topic == 'gps':
                logging.debug('Received GPS signal')
                #tz = pytz.utc # Timezone, UTC (-2 hours compared to summertime GER)
                tz = pytz.timezone('Europe/Berlin') # Timezone Bremen (UTC +2 hours)
                logging.debug('Timezone was set.')
                datetime_object = dt.datetime.fromtimestamp(float(data[0]), tz=tz)

                set_sytem_date(datetime_object)

                time_raw = datetime_object.isoformat(sep='_', timespec='seconds')
                time_parsed = time_raw.replace(':', '-')
                logging.info(f'Received time: {time_raw}')
                logging.debug(f'Time after parsing is: {time_parsed}')

                # Checking whether to start recording (only if between 5 AM and 10 PM bzw. UTC 3-20)
                if is_now_in_time_period(dt.time(5, 0), dt.time(22, 00), datetime_object.time()):
                    logging.info(f'Recording starts now for {default_video_duration/1000} seconds')
                    recording(height=default_video_height,
                              width=default_video_width,
                              framerate=default_video_framerate,
                              duration=default_video_duration,
                              filename=f'{time_parsed}')
                else:
                    logging.info('Sleeping now for 900 s')
                    time.sleep(900)
                    logging.info('Done with sleeping')
            continue
    except:
        logging.exception("Exception occurred")

if __name__ == '__main__':
    main()
