conda create --name drlnd python=3.6
source activate drlnd
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
python -m ipykernel install --user --name drlnd --display-name "drlnd"
git clone https://github.com/Nishanth009/Udacity-Nano-Degree-RL-Project-Navigation
cd Udacity-Nano-Degree-RL-Project-Navigation
wget "https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip"
unzip VisualBanana_Linux.zip
wget "https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip"
unzip VisualBanana.app.zip
wget "https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip"
unzip VisualBanana_Windows_x86.zip
wget "https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip"
unzip VisualBanana_Windows_x86_64.zip

