import cv2
from keras.models import load_model
import numpy as np

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, width + x_off, y - y_off, height + y_off)

def gender(bgr_image,x1,y1,x2,y2):
    # parameters for loading data and images
    gender_model_path = './model-weights/Gender.hdf5'
    gender_labels = {0: 'woman', 1: 'man'}
    # hyper-parameters for bounding boxes shape
    gender_offsets = (25, 25)

    # loading models
    gender_classifier = load_model(gender_model_path, compile=False)

    # getting input model shapes for inference
    gender_target_size = gender_classifier.input_shape[1:3]

    # starting lists for calculating modes
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces=(x1,y1,x2,y2)
    x1, x2, y1, y2 = apply_offsets(faces, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]
    try:
        rgb_face = cv2.resize(rgb_face, (gender_target_size))
    except:
        pass
    finally:
        rgb_face = np.expand_dims(rgb_face, 0)
        rgb_face = preprocess_input(rgb_face, False)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_probability = np.max(gender_prediction)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]
        return (gender_text,'{:.2f}'.format(gender_probability*100))
        
