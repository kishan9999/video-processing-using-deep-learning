# single_inference.py
from models import gender_model
from models import indoor_outdoor_model
from video_info import get_video_info
import cv2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run single inference on a video.')
    parser.add_argument('--video_path', type=str, help='Path to the video file.')
    args = parser.parse_args()

    #place your file here
    path = args.video_path

    # Importing Models Weights
    face1 = "./weights/opencv_face_detector.pbtxt"
    face2 = "./weights/opencv_face_detector_uint8.pb"
    gen1 = "./weights/gender_deploy.prototxt"
    gen2 = "./weights/gender_net.caffemodel"
    weights = "./weights/idod.weights.h5"


    # Load Gender and Indoor Outdoor models
    gender = gender_model(face2, face1, gen2, gen1)
    idod = indoor_outdoor_model(weights)

    #Get Video Info
    duration_sec, fps, width, height, frame_count = get_video_info(path)

    # Open the video file
    cap = cv2.VideoCapture(path)

    # Read until the video is completed
    frame_count = 0
    frame_skip_interval = fps
    res = []
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if frame_count % frame_skip_interval == 0:
            #Dectect Human and gender
            res1 = gender.find_gender(frame)
            for i in res1:
                if i not in res:
                res.append(i)

            #Predict Environment
            if frame_count == frame_skip_interval:
                image = cv2.resize(frame, (112, 112))
                res2 = idod.find_environment(image)
        else:
            break

    results = {'Title':path.split('/')[-1].split('_')[1],'Duration':duration_sec, 'FPS':fps, 'Width':width, 'Height':height, 
                    'Frames':frame_count,'Human':res, 'Environment':res2,'File Name':path.split('/')[-1]}

    # Display Results
    for key,res in results.items():
    print(key,"=", res)
    
