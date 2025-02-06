sudo apt update && sudo apt upgrade
cd app

echo "**********Installing Python 3.10 using pyenv**********"
curl https://pyenv.run | bash
if [ -e ~/.bash_profile ]
then
	printf 'export PYENV_ROOT="$HOME/.pyenv"\n[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"\neval "$(pyenv init - bash)"\n' >> ~/.bash_profile
else
	printf 'export PYENV_ROOT="$HOME/.pyenv"\n[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"\neval "$(pyenv init - bash)"\n' >> ~/.profile
	printf 'export PYENV_ROOT="$HOME/.pyenv"\n[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"\neval "$(pyenv init - bash)"\n' >> ~/.bashrc
fi
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
exec $SHELL
sudo apt install build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

pyenv install 3.10
pyenv global 3.10

# <!-- wget -qO - https://raw.githubusercontent.com/tvdsluijs/sh-python-installer/main/python.sh | sudo bash -s 3.10.0  -->
# <!-- Note that the above is also available as python.sh in /etc if you need it and it's no longer hosted. What I actually did was a simple wget and then sudo bash python.sh 3.10.0
# Also note that it's possible this command is kind of broken? pyenv Just Works but the overhead might be a bit exxy. We'll see how it runs on a pi and if we need I'll port everything to C++
# which should be less integration hell because it's all IN C++ anyway, I'm just using a wrapper function bc I love to waste time -->

echo "**********Creating venv**********"
python -m venv env
env/scripts/activate

echo "**********Setting up env for pypupilext**********"
pip install pillow==9.0
pip install numpy==1.26.4 matplotlib==3.8.4 pandas==2.2.2
pip install opencv-python==4.10.0.84

echo "**********downloading pypupilEXT**********"
wget https://github.com/openPupil/PyPupilEXT/releases/download/v0.0.1-beta/PyPupilEXT-0.0.1-cp310-cp310-linux_x86_64.whl
pip install PyPupilEXT-0.0.1-cp310-cp310-linux_x86_64.whl

echo "**********fixing "libunwind.so.1 not being found**********"
sudo apt-get install -y libunwind-dev
sudo apt-get update
sudo apt-get install -y libc++-dev

# cd ../
# git clone https://github.com/oldhorizons/counterpilot_pupil_tracking app
# cd app


