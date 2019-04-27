Development Environment:

    Windows 10 Pro, Version 1809, Build 17763.437
	  CUDA 10.0.130_411.31
	  cuDNN 7.5.1.10 for CUDA 10.0
	  Nvidia GeForce Ready Game Driver, Version 430.39
	  Python 3.7.3 (64-bit)
		    pip 19.1
		    numpy 1.16.3
		    Pillow 6.0.0
		    TensorFlow-GPU 1.13.1


Installation Instructions:

    1. Install CUDA

    2. Install cuDNN
	      https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows

    3. Install Nvidia GeForce Driver

    4. Install Python
	      -At the first screen, select "Add Python 3.7 to PATH"
	      -Select "Customize Installation", leave everything else default.

    5. Install/Update Python Packages
	      -Update pip: python -m pip install -U pip
	      -Install numpy: python -m pip install -U numpy
	      -Install Pillow: python -m pip install -U pillow
	      -Install TensorFlow-GPU: python -m pip install -U tensorflow-gpu

    6. Fix the missing cupti64_100.dll issue.
	      -Copy cupti64_100.dll from: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64
	      -Paste it in: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
	      -Copy cupti.lib from: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64
	      -Paste it in: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64

    7. Continue to the Quick Start Guide!
