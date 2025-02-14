# CounterpilotAwe
This repository implements a custom computer vision library primarily based around openCV to track the location and size of a person's pupils as they interact with the space

# TO DO
## Testing
- test people with glasses, different ethnicities, video in different lighting conditions

## Functionality
- implement OSC

## Speed
- crop images at pi end
- port to docker container
- Implement tracking rather than just detecting at every frame (see https://www.guidodiepen.nl/2017/02/detecting-and-tracking-a-face-with-python-and-opencv/) - dlib correlation tracker

## Nice to have Functionality
- Port to C++ for improved speed
- check if the iterative tracking is actually faster than just tracking the eye and then the pupil in the eye. It prooobably is but just to be sure.

# Need more info
- do we need it to run faster? I'll implement dlib if we do

# At the end
- Code cleanup. Remove comments and unnecessary memory bloat.

# SETTING UP THE OS
NB this is out of date with the new server/client system.
1. Run setup as normal. I'm using this image https://www.raspberrypi.com/software/
2. Clone this repo into the /app folder using '''git clone https://github.com/oldhorizons/counterpilot_pupil_tracking app'''
3. Run pi_setup.sh or ubuntu_setup.sh using '''sudo bash ./app/setup/pi_setup.sh'''

# CHECKING THE CAMERA WORKS
on Pi OS, run "libcamera-still" to capture and show an image, or "libcamera-vid -t 0" to show live video capture with no timeout