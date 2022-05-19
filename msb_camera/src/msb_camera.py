import picamera
import datetime as dt
import pytz, zmq, sys, logging, pickle, json
from time import sleep

# Vorlage (Code von Andreas): https://github.com/ahaselsteiner/msb-mqtt/blob/cb695298f4c4df65e5df45c128beecb90e682457/src/msb_mqtt.py

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

#todo:
def recording(height, width, framerate, duration, filename):
    file = f'Im about to be a {duration} ms long video returned as {filename}'
    return file


# Uhrzeit vom GPS Sender erhalten

# Setup
# Receive data using ZMQ.
ipc_protocol = 'tcp://127.0.0.1'
ipc_port = '5556'
connect_to = f'{ipc_protocol}:{ipc_port}'
print(f'Trying to bind zmq to {connect_to}')


try:
    while True:
        # (zmq_topic, data) = zmq_socket.recv_multipart()
        # zmq_topic = zmq_topic.decode('utf-8')
        # data = pickle.loads(data)

        zmq_topic, data = receivingData(ipc_protocol, ipc_port)

        if zmq_topic == 'gps':
            tz=pytz.utc # Timezone, UTC (-2 hours compared to summertime GER)
            tz=pytz.timezone('Europe/Berlin') # Timezone Bremen, (UTC +2 hours)
            datetime_object = dt.datetime.fromtimestamp(float(data[0]), tz=tz)
            time = datetime_object.isoformat(sep='_', timespec='seconds')
            print(f'Time received is: {time}')

            # Checking whether to start recording (1. not between 5 AM and 10 PM bzw. UTC 3-20)
            if isNowInTimePeriod(dt.time(3,0), dt.time(20,0), datetime_object.time()):
                # todo: start recording für 15 min, using timestamp as filename
                print(f'Recording right now for 5 sec')
                print(recording(height=1080, width=1920, framerate=12, duration=900000, filename=f'{time}.h264'))
                sleep(5)
            else:
                continue
        continue
except KeyboardInterrupt:
    print('Interrupted!')
    pass

# Kamera ansteuern

cam = picamera.PiCamera()
cam.resolution = (1920, 1080)

# 1. Versuch ein Bild zu erstellen

cam.capture('bild.jpg')
cam.close()

# 2. Versuch ein Video zu erstellen

cam = picamera.PiCamera()
try:
   cam.resolution = (1920, 1080)
   cam.start_recording('filmchen.h264')
   cam.wait_recording(30) #30 Sekunden Aufnahme
   cam.stop_recording()
finally:
   cam.close()


# 3. Versuch Kommandozeile
import os
os.system("libcamera-vid -n --height 1080 --width 1920 --framerate 12 -o test_1080_1920.h264 --save-pts timestamps_1080_1920.txt")
#os.system("raspivid -w 1920 -h 1080 -fps 90 -t 30000 -o myvid.h264")
