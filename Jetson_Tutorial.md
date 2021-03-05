# Jetson Tutorial
This tutorial will be helpful if you aleardy have your model, which is about how to optimize the model and deploy the model on Jetson.

### Step 1: Model optimization
It is more efficient to optimize the model using your PC than Jetson.

1. **For Pytorch models**, the process is: 'origianl Pytorch model->ONNX model->TensorRT model'

Converting the origianl model to ONNX model, you need just one instruction 'torch.onnx.export', which requires the pre-trained model itself, tensor with the same size as input data, name of ONNX file, input and output names:
```python
torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=['input'],
                  output_names=['output'], export_params=True)
```
You can also visualize the ONNX model using Netron: type Netron in the command line and open http://localhost:8080/ at your browser. <br>
To install Netron:
```python
python3 -m pip install netron
```
Converting the ONNX model to TensorRT model, you need to parse the ONNX model and initialize TensorRT Context and Enginer. You will need to create a Builder and then the builder can create Network and generate Engine (that would be optimized) from this network. Next, with Network definition you can create an instance of Parser. Finally, parse our ONNX file.
```python
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')
```
After that, we can generate the Engine and create the executable Context. The engine takes input data, performs inferences, and emits inference output.
```Python
# generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context

```

2. **For Tensorflow models**, the process is: 'original Tensorflow model->TensorRT model'

TensorRT can be directly called from Tensorflow models, thus it is very convenient to use TensorRT to analyze the TensorFlow graph and apply optimizations. You need tensor inputs as 'my_input_fn' and output directory as 'output_saved_model_dir'. The converter should be construct before converting, there are several parameters for a converter: <br>
frozen_graph_def: frozen TensorFlow graphout <br>
put_node_name: list of strings with names of output nodes e.g. “resnet_v1_50/predictions/Reshape_1”] <br>
max_batch_size: integer, size of input batch e.g. 16 <br>
max_workspace_size_bytes: integer, maximum GPU memory size available for TensorRT <br>
precision_mode: string, allowed values “FP32”, “FP16” or “INT8” <br>

```python
import numpy as np
from functools import partial
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS 
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(maximum_cached_engines=100)
converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir, conversion_params=conversion_params)
converter.convert()

converter.build(input_fn =my_input_fn)
converter.save(output_saved_model_dir)
```
TensorRT takes a frozen TensorFlow graph as input and returns an optimized graph with TensorRT nodes. Also, you can visualize the converted TensorRT model:
```python
!saved_model_cli show --all --dir output_saved_model_dir
```

### Step 2: Save files at Jetson locally
You should transfer the final model from your PC to Jetson first. One recommended way is to upload all the files to Google Cloud, then you can login your google account in Jetson browser and download the files to Jetson.

### Step 3: Create the required environment on Jetson
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

6. **torchvision**
```
sudo apt-get install libjpeg-dev zlib1g-dev  #install dependencies libjpeg-dev and zlib1g-dev
git clone --branch <version> https://github.com/pytorch/vision torchvision  #Clone Torchvision e.g. v0.3.0
cd torchvision  #install Torchvision
sudo python setup.py install
(or simply - python setup.py install - if you aren't the root user)
cd ../
```
you can use 'import torchvision ' to verify

7. **OpenCV** <br>
OpenCV for both python2 and python3 is pre-installed on Jetson. If you encountered some problem importing OpenCV, you can find more information through this link: (https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html)

8. **ONNX**
```
sudo apt-get install protobuf-compiler libprotoc-dev  #if there is no anaconda environment, you need to install dependencies
sudo pip3 install onnx==1.4.1  #install
```


9. **Archiconda** <br>
Jetson runs on a AArch64 architecture, so the official version of anaconda is unavailable. Instead, you need to install Archiconda.
```
wget --quiet -O archiconda.sh https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh && \
    sh archiconda.sh -b -p $HOME/archiconda3 && \
    rm archiconda.sh
```
more information please view: (https://forums.developer.nvidia.com/t/anaconda-for-jetson-nano/74286)

10. **Virtual environment** <br>
```
source ~/archiconda3/bin/activate root
conda env list  #print the list of all virtual environments
conda create --name env_name  #create a new virtual environment
conda actviate env_name  #activate the virtual environment
conda deactivate  #deactivate the current environment
conda remove --name env_name --all  #delete the environment
```
