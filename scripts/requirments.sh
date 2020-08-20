echo "Installing Python libraries \n"
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install sklearn
pip3 install sklearn_crfsuite
pip3 install tqdm==4.46.1
pip3 install tensorflow
pip3 install Keras
pip3 install git+https://www.github.com/keras-team/keras-contrib.git
#pip3 install git+https://git@github.com/cdli-gh/pyoracc.git@master#egg=pyoracc
pip3 install OpenNMT-py
pip3 install click
pip3 install nltk
pip3 install flair
pip3 install tokenizers
pip3 install transformers
echo -e "\n Downloading Backward Translation Model (Best Performing)\n"
wget https://cdlisumerianunmt.s3.us-east-2.amazonaws.com/BackTranslation/5st/_step_10000.pt -O Translation_Models/Back_Translation.pt
