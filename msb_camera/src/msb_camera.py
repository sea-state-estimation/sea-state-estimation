import datetime as dt
import pytz, zmq, sys, logging, pickle
import os

# Vorlage für Sockets und Datenempfang (Code von Andreas): https://github.com/ahaselsteiner/msb-mqtt/blob/cb695298f4c4df65e5df45c128beecb90e682457/src/msb_mqtt.py
# Vorlage für isNowInTimePeriod (StackOverflow, 19.05.2022): https://stackoverflow.com/questions/10048249/how-do-i-determine-if-current-time-is-within-a-specified-range-using-pythons-da

def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else:
        #Over midnight:
        return nowTime >= startTime or nowTime <= endTime

def receivingData(ipc_protocol, ipc_port):
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

    #print('Successfully bound to zeroMQ receiver socket as subscriber')
    #print('Trying to receive data.')

    (zmq_topic, data) = zmq_socket.recv_multipart()
    zmq_topic = zmq_topic.decode('utf-8')
    data = pickle.loads(data)
    return zmq_topic, data

'''start camera for recording a sequence, 
see official documentation of libcamera-vid: https://www.raspberrypi.com/documentation/accessories/camera.html#libcamera-vid'''

def recording(height, width, framerate, duration, filename):
    # todo: include filepath for designated storage (usb stick via Hub)
    command = f'libcamera-vid -n --height {height} --width {width} --framerate {framerate} -o {filename}.h264 --save-pts {filename}_timestamps.txt'
    os.system(command)

###################### START SCRIPT ######################
# 1. Receiving date and time from gps sensor

# Setup
# Receive data using ZMQ.
ipc_protocol = 'tcp://127.0.0.1'
ipc_port = '5556'
connect_to = f'{ipc_protocol}:{ipc_port}'
print(f'Trying to bind zmq to {connect_to}')

try:
    while True:
        zmq_topic, data = receivingData(ipc_protocol, ipc_port)

        if zmq_topic == 'gps':
            tz=pytz.utc # Timezone, UTC (-2 hours compared to summertime GER)
            tz=pytz.timezone('Europe/Berlin') # Timezone Bremen, (UTC +2 hours)
            datetime_object = dt.datetime.fromtimestamp(float(data[0]), tz=tz)
            time = datetime_object.isoformat(sep='_', timespec='seconds')
            print(f'Time received is: {time}')

            # Checking whether to start recording (only if between 5 AM and 10 PM bzw. UTC 3-20)
            if isNowInTimePeriod(dt.time(5,0), dt.time(22,0), datetime_object.time()):
                #start recording for 5 sec / 15 min, using timestamp as filename
                duration = 5000  # for testing short duration of 5 s = 5000 ms, final duration supposed to be 15 min = 900000 ms
                print(f'Recording right now for {duration/1000} s')
                recording(height=1080, width=1920, framerate=12, duration=duration, filename=f'{time}')
            else:
                continue
        continue
except KeyboardInterrupt:
    print('Interrupted!')
    pass
