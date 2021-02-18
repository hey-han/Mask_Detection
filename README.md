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

1. **Ensure all packages are up to date**
```python
sudo apt-get update #important step before the installation of new packages
```

2. **pip3**

'''
sudo apt-get install python3-pip  #install
pip3 --version  #check pip version
pip3 --help  #view the list of all pip commands and options
'''
3. **Python3**
'''

'''

4. **Pytorch**
'''



