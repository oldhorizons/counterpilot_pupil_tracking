# CounterpilotAwe
This repository implements a custom computer vision library primarily based around openCV to track the location and size of a person's pupils as they interact with the space

# TO DO
## Testing
- test people with glasses, different ethnicities, video in different lighting conditions

## Functionality
- server / communications wrappers. Check protocol.
- Port to a linux container
- Implement on a raspberry pi and test speed
- Implement capture from video feed https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
- Implement tracking rather than just detecting at every frame (see https://www.guidodiepen.nl/2017/02/detecting-and-tracking-a-face-with-python-and-opencv/) 
- Iris tracking
### Iris Tracking
- https://github.com/openPupil/PyPupilEXT THIS IS THE ONEEE
- https://github.com/openPupil/Open-PupilEXT - GUI unnecessary in terms of bloat. See if you can find wrappers for the libraries below:
- https://github.com/thirtysixthspan/cvEyeTracker/tree/master (C/C++ - GNU general public license)
- https://github.com/LeszekSwirski/singleeyefitter/tree/master (C, MIT license - no restrictions)
- https://github.com/tcsantini/EyeRecToo/tree/master (C++/C) - implements EyeRecToo, ElSe, PuRe, PuReSt

## Nice to have Functionality
- Port to C++ for improved speed
- check if the iterative tracking is actually faster than just tracking the eye and then the pupil in the eye. It prooobably is but just to be sure.

# Need more info
- where are the cameras? what angle will the face be at? how much of the face will be visible?
- headset/brainwave implementation. check emails for SDK access

# At the end
- Code cleanup. Remove comments and unnecessary memory bloat.



# SETTING UP THE OS: RASPBERRY PI OS
1. Run setup as normal. I'm using this image https://www.raspberrypi.com/software/
2. Run the following commands, in order:
'''
sudo apt update && sudo apt upgrade
mkdir app | cd app
<!-- switch to python 3.10.0 -->
wget -qO - https://raw.githubusercontent.com/tvdsluijs/sh-python-installer/main/python.sh | sudo bash -s 3.10.0 
<!-- Note that the above is also available as python.sh in /etc if you need it and it's no longer hosted. What I actually did was a simple wget and then sudo bash python.sh 3.10.0-->

<!-- set up env for pypupilext -->
pip install pillow==9.0
pip install numpy==1.26.4 matplotlib==3.8.4 pandas==2.2.2
pip install opencv-python==4.10.0.84

wget https://github.com/openPupil/PyPupilEXT/releases/download/v0.0.1-beta/PyPupilEXT-0.0.1-cp310-cp310-linux_x86_64.whl

'''



# SETTING UP THE OS V2: UBUNTU / GENERIC
using the latest release of ubuntu desktop
'''
sudo snap install curl
curl -fsSL https://pyenv.run | bash
sudo snap install git
<!-- FROM https://github.com/pyenv/pyenv?tab=readme-ov-file#a-getting-pyenv -->
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init - bash)"' >> ~/.profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init - bash)"' >> ~/.bash_profile
exec "$SHELL"

<!-- https://github.com/pyenv/pyenv/wiki#suggested-build-environment -->
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

pyenv install 3.10
pyenv global 3.10

<!-- set up env for pypupilext - https://github.com/openPupil/PyPupilEXT?tab=readme-ov-file -->
pip install pillow==9.0
pip install numpy==1.26.4 matplotlib==3.8.4 pandas==2.2.2
pip install opencv-python==4.10.0.84

wget https://github.com/openPupil/PyPupilEXT/releases/download/v0.0.1-beta/PyPupilEXT-0.0.1-cp310-cp310-linux_x86_64.whl
pip install PyPupilEXT-0.0.1-cp310-cp310-linux_x86_64.whl

sudo mkdir app




FOR DEBUGGING PURPOSES ONLY
sudo snap install --classic code

'''