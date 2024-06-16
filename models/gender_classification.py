import cv2 

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) 

class gender_model():
    def __init__(self,face2, face1, gen2, gen1) -> None:
        # Using models 
        self.face = cv2.dnn.readNet(face2, face1)  # Face 
        self.gen = cv2.dnn.readNet(gen2, gen1)   # gender 
        self.lg = ['Male', 'Female']  # Categories of distribution 

    def find_gender(self,image):
        # Copy image 
        fr_cv = image.copy()

        # Face detection 
        fr_h = fr_cv.shape[0] 
        fr_w = fr_cv.shape[1] 
        blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300, 300), [104, 117, 123], True, False) 
        
        self.face.setInput(blob) 
        detections = self.face.forward()

        # Face bounding box creation 
        faceBoxes = [] 
        gender_info = set()
        for i in range(detections.shape[2]): 
            #Bounding box creation if confidence > 0.7
            confidence = detections[0, 0, i, 2] 
            if confidence > 0.7: 
                x1 = int(detections[0, 0, i, 3]*fr_w)
                y1 = int(detections[0, 0, i, 4]*fr_h)
                x2 = int(detections[0, 0, i, 5]*fr_w)
                y2 = int(detections[0, 0, i, 6]*fr_h)
                faceBox = [x1, y1, x2, y2]
                
                #Extracting face as per the faceBox 
                faceimg = fr_cv[max(0, faceBox[1]-15):
                            min(faceBox[3]+15, fr_cv.shape[0]-1),
                            max(0, faceBox[0]-15):min(faceBox[2]+15,
                                                    fr_cv.shape[1]-1)]
                
                #Extracting the main blob part
                blob = cv2.dnn.blobFromImage(faceimg, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                #Prediction of gender
                self.gen.setInput(blob)
                genderPreds = self.gen.forward()
                gender = self.lg[genderPreds[0].argmax()] 
                gender_info.add(gender)
                # print(gender)
        return sorted(gender_info)


if __name__ == "__main__":
    print("hello")