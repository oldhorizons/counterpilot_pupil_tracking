#run from inside the target repo. Assumes python and pip already installed
#set up env
# python -m venv env
# source env/bin/activate

sudo apt install python3-picamzero

# sudo apt-get install libatlas-base-dev libavcodec-dev libavformat-dev libgtk-3-dev libopenjp2-7 libswscale-dev python3-pip
# # pip3 install numpy opencv-python picamera2
# sudo apt install libcap-dev python3-libcamera #python3-picamzero python3-picamera2??

# pip install opencv-python #picamera2[array]

#set up the rpi camera for cv2 processing: https://www.raspberrypi.org/forums/viewtopic.php?f=43&t=62364 - may not need?
# sudo modprobe bcm2835-v4l2


#print IP address
echo ifconfig

# NOTE: https://forums.raspberrypi.com/viewtopic.php?t=369326 - DO NOT USE A VENV