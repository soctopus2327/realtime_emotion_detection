from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from keras.models import load_model

app = Flask(__name__)
CORS(app) 

model = load_model('Final_Emotion_Detection_Model_v1.h5')

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image')

        img = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        results = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_roi = img[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (224, 224))  
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_roi = face_roi / 255.0
                face_roi = np.reshape(face_roi, (1, 224, 224, 1))

                predictions = model.predict(face_roi)
                emotion_index = np.argmax(predictions)
                predicted_emotion = emotion_labels[emotion_index]
                results.append({'emotion': predicted_emotion, 'coordinates': (x, y, w, h)})
                print("Detected emotion:", predicted_emotion)
                print(predictions) 

        else:
            print("No faces detected")

        return jsonify(results)
    except Exception as e:
        print("Error in prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
