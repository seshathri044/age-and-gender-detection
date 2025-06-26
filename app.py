from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image
import your_model_code  # Import your model code here for processing

app = Flask(__name__)
socketio = SocketIO(app)

# Route for serving the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # The HTML you provided earlier

# WebSocket route for video feed
@socketio.on('message')
def handle_video_frame(message):
    if 'image' in message:
        # Decode the base64 image
        img_data = message['image'].split(",")[1]  # Remove base64 prefix
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))

        # Call your model code here to get predictions (age, gender)
        gender, age = your_model_code.predict(img)  # Assuming `your_model_code` has a `predict` function
        
        # Send the results back to the frontend
        emit('result', {'gender': gender, 'age': age})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)






























# from flask import Flask, render_template
# from flask_socketio import SocketIO, emit
# import cv2
# import base64
# import numpy as np
# from io import BytesIO
# from your_model_code import detect_gender_and_age # import your gender/age detection function

# app = Flask(__name__)
# socketio = SocketIO(app)

# @app.route('/')
# def index():
#     return render_template('index.html')  # The HTML frontend you provided

# @socketio.on('message')
# def handle_message(msg):
#     print('Received message:', msg)

# @socketio.on('json')
# def handle_json(data):
#     # Get the base64 image from the frontend
#     image_data = data['image']
#     img_data = base64.b64decode(image_data.split(',')[1])  # Remove the "data:image/jpeg;base64," part

#     # Convert to a numpy array
#     np_arr = np.frombuffer(img_data, np.uint8)
#     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     # Run gender and age detection
#     gender, age = detect_gender_and_age(img)

#     # Send back the results
#     emit('message', {'gender': gender, 'age': age})

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=5000)
