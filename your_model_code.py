import cv2
import numpy as np

# Load the pre-trained models for gender and age detection
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# List of possible age ranges
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# List of possible genders
genderList = ['Male', 'Female']

# Function to detect age and gender
def detect_gender_and_age(image):
    # Prepare the image for the neural network
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    
    # Gender prediction
    genderNet.setInput(blob)
    genderPred = genderNet.forward()
    gender = genderList[genderPred[0].argmax()]  # Get the predicted gender

    # Age prediction
    ageNet.setInput(blob)
    agePred = ageNet.forward()
    age = ageList[agePred[0].argmax()]  # Get the predicted age

    return gender, age
