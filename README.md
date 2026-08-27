# Generator-Decoder Graph Attention Networks For 3D Hand Pose Reconstruction

This project aims to correct distortions seen in naive 3D stereo camera projection of the 21 hand keypoints by (1) creating a custom stereo dataset to describe and label the distortions present in hand-tracking, (2) train a model via a new neural network architecture that leverages generative modelling and graph attention networks along with the STB-Pose dataset to correct these errors, and (3) visually observe the results to see if the pose and positioning is correct.

## Installation

1. Clone Project
```
git clone https://github.com/mtgith/Generator-Decoder-GAT-Hand-Tracking.git
```

2. Setup Virtual Environment
```
conda create -n modelenv python=3.10
conda activate modelenv
```

3. Install Requirements
```
pip install -r requirments.txt
```

4. Training the Model
To train the model, install the STB-Pose dataset, along with our self-constructed dataset from the Google Drive link: . After the dataset is installed, create a MANO account to install the mano_v1_2.zip file for the MANO_RIGHT template. After this, create a directory mano/models to alter the MANO path in the json configuration files. Once complete, begin the pretraining stage by exectuing pretrain.py. Ensure, that the mode is "TRAIN" not "OPTUNA". After this is done, execute posttrain.py, and lastly manotrain.py. All models were trained on Ubuntu with an RTX 3090, so install the appropriate CUDA Driver and PyTorch version for your machine.

It is important to note that inference can be done without training, but if you do decide to take the full training route, one must filter the dataset with the filtering files in the folder directory, and create the dataset using the same method done in the STB-Pose dataset by using temporal frame images, and using the optimizedataset.py file then the graphdataset.py file and matrixnorm.py file to obtain the required files for training.

Note that training the model is entirely unnecessary though, as the PCA space in MANO should be sufficiently large to cover most hands, and one can simply import the model weights to immediately perform inference. 

5. Inference
To perform inference on a video, take your stereo camera, record a video of your hand (MANO only uses right, so if you want to use left, you must mirror the coordinates around the wrist axis before showing the plot), and then apply Umeyama transformation on the final plot in videval.py to transform from our camera space to yours (note that this is only necessary if you didn't train with your own dataset).

For photo inference, simply take a stereo phot of your hand and Umeyama transform into your space to get accurate keypoint locations in 3D space. Note that the error depends on your camera intrinsics and is entirely dependent on your reprojection error. In our tests we obtained a MPJPE of 26.3mm. 

For real-time tracking, one can simply take the videval.py and apply a simple tracking pipeline by processing each frame in real time.
