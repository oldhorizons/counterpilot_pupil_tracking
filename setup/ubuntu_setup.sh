sudo snap install curl
curl -fsSL https://pyenv.run | bash
sudo snap install git
# <!-- FROM https://github.com/pyenv/pyenv?tab=readme-ov-file#a-getting-pyenv -->
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

# <!-- https://github.com/pyenv/pyenv/wiki#suggested-build-environment -->
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

pyenv install 3.10
pyenv global 3.10

cd app

# <!-- set up env for pypupilext - https://github.com/openPupil/PyPupilEXT?tab=readme-ov-file -->
pip install pillow==9.0
pip install numpy==1.26.4 matplotlib==3.8.4 pandas==2.2.2
pip install opencv-python==4.10.0.84

wget https://github.com/openPupil/PyPupilEXT/releases/download/v0.0.1-beta/PyPupilEXT-0.0.1-cp310-cp310-linux_x86_64.whl
pip install PyPupilEXT-0.0.1-cp310-cp310-linux_x86_64.whl
# <!-- if libunwind.so.1 can't be found -->
sudo apt-get install -y libunwind-dev
sudo apt-get update
sudo apt-get install -y libc++-dev

pip install python-osc