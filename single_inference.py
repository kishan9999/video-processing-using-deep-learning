from models import gender_model
from models import indoor_outdoor_model

import cv2

# Importing Models Weights
face1 = "./weights/opencv_face_detector.pbtxt"
face2 = "./weights/opencv_face_detector_uint8.pb"
gen1 = "./weights/gender_deploy.prototxt"
gen2 = "./weights/gender_net.caffemodel"
weights = "./weights/idod.weights.h5"


gender = gender_model(face2, face1, gen2, gen1)

# Input image 
image = cv2.imread(r'F:\Live Project\2023 Cloud API\project 15624\output1\birthday-3016615_640.jpg') 
res = gender.find_gender(image)

print(res)


idod = indoor_outdoor_model(weights)
