# Video Processing Using Deep Learning

## Table of Contents
- [Introduction](#introduction)
- [How to Use](#how-to-use)
- [Documentation](#documentation)
- [License](#license)
- [Authors](#authors)
- [References](#references)

## Introduction
This repository utilizes deep learning models for video processing tasks, focusing on face detection, gender classification, and indoor/outdoor scene classification. It includes implementations using OpenCV and pretrained models such as MobileNetV2.

### Face Detection and Gender Classification
![Face Detection and Gender Classification](https://github.com/kishan9999/video-processing-using-deep-learning/blob/main/samples/demo.png)

For face detection and gender classification, OpenCV and deep learning models were employed. The models achieve significant accuracy in identifying faces and classifying genders in images and video streams.

checkout this notebook for demo [Face detection and gender classification.ipynp](https://github.com/kishan9999/video-processing-using-deep-learning/blob/main/human%20detection%20and%20gender%20classification%20demo.ipynb)

### Indoor/Outdoor Scene Classification
Using a pretrained MobileNetV2 model on the SUN dataset, this project performs indoor/outdoor scene classification. The model was trained on 60,000 images, achieving high accuracy rates:

<div>
    <img src="https://github.com/kishan9999/video-processing-using-deep-learning/blob/main/samples/mobilenetv2%20training.png" alt="training 1" width="400"/>
    <img src="https://github.com/kishan9999/video-processing-using-deep-learning/blob/main/samples/mobilenetv2%20training%202.png" alt="training 2" width="400"/>
</div>

- Training Accuracy: 95.25%
- Validation Accuracy: 93.33%
- Testing Accuracy: 93.66%
  
Complete training details can be found in the Jupyter Notebook [training classification model.ipynb](https://github.com/kishan9999/video-processing-using-deep-learning/blob/main/training%20cnn%20model%20for%20indoor%20outdoor%20scenes.ipynb)

### Video Processing using Jupyter Notebook
Here, 1000 videos have been tested for human detection, gender classification, indoor/outdoor classification, and also to provide video information such as resolution, frame rates, and duration.
[testing on 1000 videos.ipynb](https://github.com/kishan9999/video-processing-using-deep-learning/blob/main/processing%20all%201000%20videos.ipynb)

Model weights are available [(idod.weights.h5)](https://github.com/kishan9999/video-processing-using-deep-learning/blob/main/weights/idod.weights.h5).

## How to Use
To use this repository, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/kishan9999/video-processing-using-deep-learning.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure the following packages are installed:
   - pandas==2.0.3
   - tensorflow==2.15.0
   - opencv-python==4.8.0.76
   - numpy==1.25.2
   - matplotlib==3.7.1
   - flask==2.2.5
  
3. Download All weights and place it in the weights folder
   * [gender_deploy.prototxt](https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/gender_deploy.prototxt)
   * [gender_net.caffemodel](https://github.com/smahesh29/Gender-and-Age-Detection/blob/master/gender_net.caffemodel)
   * [opencv_face_detector_uint8.pb](https://github.com/spmallick/learnopencv/blob/master/AgeGender/opencv_face_detector_uint8.pb)
   * [opencv_face_detector.pbtxt](https://github.com/spmallick/learnopencv/blob/master/AgeGender/opencv_face_detector.pbtxt)

5. For video processing, use `single_inference.py`:
   ```bash
   python single_inference2.py --video_path
   ```
   OR replace `path` with your video file in python single_inference.py.

## Video INFO API 
run this code [video_info.py](https://github.com/kishan9999/video-processing-using-deep-learning/blob/main/video_info.py) to display information about the video file. 
![](https://github.com/kishan9999/video-processing-using-deep-learning/blob/main/samples/video%20info%20api.png)

## Documentation
For detailed documentation, refer to [Document](https://sites.google.com/view/video-processing-deep-learning?usp=sharing)

## License
This project is licensed under the [MIT License](link/to/license).

## Author
- Kishan Joshi

## References
- [Keras Models](https://keras.io/api/applications/)
- [SUN dataset](https://groups.csail.mit.edu/vision/SUN/hierarchy.html)
- [Age Gender Code Guide](https://www.geeksforgeeks.org/age-and-gender-detection-using-opencv-in-python/)
- [Age Gender Code Github](https://github.com/spmallick/learnopencv/tree/master/AgeGender)
- Dataset used for action recognition: [UCF101 Action Recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition)
