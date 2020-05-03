from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np

def emotion(faces,x1, y1, x2, y2):
    # parameters for loading data and images
    emotion_model_path = './model-weights/_mini_XCEPTION.102-0.66.hdf5'

    # hyper-parameters for bounding boxes shape
    # loading models
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

    gray = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    #faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    roi = gray[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
        
        
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]


    return (label,"{:.2f}".format(emotion_probability*100))


                

