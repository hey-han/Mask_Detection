# Mask_Detection
A real-time mask detection project using Jetson Xavier NX <br>
Contributors: Dong Sheng, Shitala Prasad, Jovan Hermawan, Zhang Han

## Jetson Tutorial
This tutorial will be helpful if you aleardy had your model and optimised it using your PC, which is about how to deploy the model on Jetson.

#### Step 1: Save files at Jetson locally
Since it is more efficient to optimise the model using your PC, you should transfer the final model from your PC to Jetson first. One recommended way is to upload all the files to Google Cloud, then you can login your google account in Jetson browser and download the files to Jetson.

#### Step 2: Create the required environment on Jetson
After transferring the model from your PC to Jetson, you may notice that some syntax does not working on Jetson, especially those about package installations. You should modify your code a bit and check which packages are required to be installed manually. 

The indtallation of some common packages and many related instructions are given here, which may help you.

1. **Ensure all packages are up to date** <br>
important step before the installation of new packages <br>
```
sudo apt-get update 
sudo apt-get upgrade
```

2. **pip3**
```
sudo apt-get install python3-pip  #install
sudo pip3 install -U pip testresources setuptools
pip3 --version  #check pip version
pip3 --help  #view the list of all pip commands and options
```

3. **Python3**
```
sudo apt install python3.7  #install
python ––version  #check python version
```

4. **Pytorch**
```
pip3 install Cython
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
pip3 install numpy torch-1.6.0-cp36-cp36m-linux_aarch64.whl  #install
```
```
import torch  #verify
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
```

5. **Tensorflow**
```
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran  #install system packages required by TensorFlow
sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow  #this command will install the latest version of TensorFlow compatible with the JetPack
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 ‘tensorflow<2’  #install a TensorFlow 1.x package
```
more details of Tensorflow installation can be found here: (https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)





