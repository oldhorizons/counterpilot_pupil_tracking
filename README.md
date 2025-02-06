# CounterpilotAwe
This repository implements a custom computer vision library primarily based around openCV to track the location and size of a person's pupils as they interact with the space

# TO DO
## Testing
- test people with glasses, different ethnicities, video in different lighting conditions
- test video + tracking
- put OS setup into a script you can just run from git pull
- find an appropriately lightweight OS

## Functionality
- server / communications wrappers. Check protocol.
- Port to a linux container
- Implement on a raspberry pi and test speed
- Implement capture from video feed https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
- Implement tracking rather than just detecting at every frame (see https://www.guidodiepen.nl/2017/02/detecting-and-tracking-a-face-with-python-and-opencv/) - dlib correlation tracker

## Nice to have Functionality
- Port to C++ for improved speed
- check if the iterative tracking is actually faster than just tracking the eye and then the pupil in the eye. It prooobably is but just to be sure.

# Need more info
- where are the cameras? what angle will the face be at? how much of the face will be visible? what communication protocol does the camera use?
- what kind of Rpi will we be using? x64 architecture? (please)
- do we need it to run faster? I'll implement dlib if we do
- headset/brainwave implementation. check emails for SDK access
- what sort of interfacing do you want to do with the software? manual setup or just track and hope for the best?  honestly template matching with an appropriately sized template might be a great start

# At the end
- Code cleanup. Remove comments and unnecessary memory bloat.

# SETTING UP THE OS
1. Run setup as normal. I'm using this image https://www.raspberrypi.com/software/
2. Clone this repo into the /app folder using '''git clone https://github.com/oldhorizons/counterpilot_pupil_tracking app'''
3. Run pi_setup.sh or ubuntu_setup.sh using '''sudo bash ./app/setup/pi_setup.sh'''