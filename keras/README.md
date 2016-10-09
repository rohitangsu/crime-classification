Install mkvirtualenv(or any other package for virtual environment creation.)

sudo pip install virtualenvwrapper

Run:
1. mkvirtualenv keras
2. workon keras


Install the following packages.
pip install numpy scipy
pip install scikit-learn
pip install pillow
pip install h5py


Install Keras for deep learning.
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
git clone https://github.com/Theano/Theano
cd Theano
python setup.py install

pip install keras

Test Installation:
	
$ workon keras
$ python
>>> import keras
>>>


To enable GPU support, try this link.
http://deeplearning.net/software/theano/tutorial/using_gpu.html
