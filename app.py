# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# import base64
# from io import BytesIO
# from PIL import Image

# app = Flask(__name__)

# # Dummy function for age and gender detection (replace with actual model)
# def predict_age_gender(image):
#     # Use a pre-trained model here for age and gender detection
#     # For simplicity, return dummy values
#     return 30, "Male"

# @app.route("/detect", methods=["POST"])
# def detect():
#     data = request.get_json()
#     frame_data = data.get("frame")
    
#     # Decode the base64-encoded image
#     img_data = base64.b64decode(frame_data.split(',')[1])
#     img = Image.open(BytesIO(img_data))
#     img = np.array(img)

#     # Convert the image to BGR format for OpenCV
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     # Perform the detection (replace this with a real model)
#     age, gender = predict_age_gender(img)

#     return jsonify({"age": age, "gender": gender})

# if __name__ == "__main__":
#     app.run(debug=True)




#  from flask import Flask, render_template, request, jsonify
# import cv2
# import numpy as np

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')  # Render the frontend page

# @app.route('/detect', methods=['POST'])
# def detect():
#     # Here, you would extract the frame from the request, process it, and get predictions
#     frame = request.get_json().get('frame')
#     # OpenCV logic to detect age and gender (example)
    
#     # Simulate prediction (replace with real OpenCV-based detection logic)
#     predicted_age = 25  # This would come from your OpenCV model
#     predicted_gender = "male"  # This would come from your OpenCV model
    
#     return jsonify({'age': predicted_age, 'gender': predicted_gender})

# if __name__ == '__main__':
#     app.run(debug=True)
# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)

# Load the models for face detection, age prediction, and gender prediction
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # Mean values for pre-trained models
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Helper function to detect faces
def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
    
    return bboxs

# Flask route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for detecting age and gender
@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    frame_data = data.get('frame')
    img_data = base64.b64decode(frame_data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Detect faces
    bboxs = faceBox(faceNet, frame)

    results = []

    for bbox in bboxs:
        face = frame[max(0, bbox[1]):min(bbox[3], frame.shape[0]), 
                     max(0, bbox[0]):min(bbox[2], frame.shape[1])]
        
        # Prepare the face for gender and age prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender Prediction
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        
        # Age Prediction
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        results.append({'age': age, 'gender': gender, 'bbox': bbox})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
