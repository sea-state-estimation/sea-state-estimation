# msb_camera
Code that is supposed to be integrated to msb in scr

## Automated video capturing on Motion Sensor Box (msb_camera)
### Hardware
Raspberry Pi Video v2.1, USB-Stick on Hub

### /msb_camera/src/msb_camera.py
Contains source code for automated video recording independent of internet connection

### /msb_camera/src/camera_config.py
Not yet used, all configurations are done in msb_camera.py

### /msb_camera/config/msb-camera.service
Calls python script msb_camera.py as a service, as intended it starts with every reboot.

