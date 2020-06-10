## Environment Setup

### 1. Pre-requisites:
- Unix system (Ubuntu)
- Python version 3
- Integrated development environment (IDE) (e.g. PyCharm or Sublime Text)


### 2. Install *virtualenv* 

`pip3 install virtualenv`

Execute
`virtualenv -p python3 --no-site-packages venv`

This installs a sandboxed Python in the directory `venv`. To use this *virtualenv* for your project, first
activate it by calling:

`source venv/bin/activate`

To test whether your *virtualenv* is activated, check using the command:

`which python`

This should now point to `venv/bin/python`.

### 3. Install required packages:

`pip3 install -r requirements.txt`

The file **requirements.txt** contains the list of all dependencies with the version specified used for this project.

<!--
#### 4. Tensorflow GPU Installation (CUDA-enabled)
`pip install tensorflow-gpu==2.2.0rc2`

`pip install tensorboard`

##### (Current release for CPU-only)
`pip install tensorflow`

`pip install tensorboard`
### GPU package for CUDA-enabled GPU cards
-->



### 4. Pytorch Installation (Linux GPU Cuda version 10.2.89)

##### Install Pytorch
`pip3 install torch==1.5.0 torchvision==0.6.0`

##### Install IPython
`pip install Ipython==7.14.0`

##### Install Tensorboard for pytorch
`pip3 install tensorboard-pytorch==0.7.1 --no-cache-dir`