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

# SETTING UP THE OS: RASPBERRY PI OS
1. Run setup as normal. I'm using this image https://www.raspberrypi.com/software/
2. Run the following commands, in order:
'''
sudo apt update && sudo apt upgrade
mkdir app | cd app
<!-- switch to python 3.10.0 -->
wget -qO - https://raw.githubusercontent.com/tvdsluijs/sh-python-installer/main/python.sh | sudo bash -s 3.10.0 
<!-- Note that the above is also available as python.sh in /etc if you need it and it's no longer hosted. What I actually did was a simple wget and then sudo bash python.sh 3.10.0
Also note that it's possible this command is kind of broken? pyenv Just Works but the overhead might be a bit exxy. We'll see how it runs on a pi and if we need I'll port everything to C++
which should be less integration hell because it's all IN C++ anyway, I'm just using a wrapper function bc I love to waste time -->

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
<!-- if libunwind.so.1 can't be found -->
sudo apt-get install -y libunwind-dev
sudo apt-get update
sudo apt-get install -y libc++-dev

sudo mkdir app
git clone https://github.com/oldhorizons/counterpilot_pupil_tracking app

'''