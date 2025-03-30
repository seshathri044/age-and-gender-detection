import cv2
import numpy as np
# Function to detect faces
def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:  # Threshold for face detection confidence
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    return frame, bboxs

# Load the face, age, and gender models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Predefined constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # Mean values for pre-trained models
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Initialize the video stream (webcam)
video = cv2.VideoCapture(0)

padding = 20  # Padding around face

while True:
    ret, frame = video.read()  # Capture frame from webcam
    if not ret:
        break
    
    # Detect faces in the frame
    frame, bboxs = faceBox(faceNet, frame)
    
    for bbox in bboxs:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1), 
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        
        # Preprocess face image for gender and age detection
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender Prediction
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]  # Get the gender with highest probability
        
        # Age Prediction
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]  # Get the age range with highest probability
        
        # Display label with gender and age
        label = f"{gender}, {age}"
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)  # Label background
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the result frame with bounding boxes and labels
    cv2.imshow("Age and Gender Detection", frame)
    
    # Exit if 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release resources and close windows
video.release()
cv2.destroyAllWindows()
