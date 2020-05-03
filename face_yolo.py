import argparse
from PIL import Image
import cv2
from face.yolo import YOLO
from detect_open_mouth import Detect_mouth_opening
from emotion_pred import emotion
from Gender_detection import gender
from eyeglass_detector import glasses


#####################################################################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model-weights/YOLO_Face.h5',
                        help='path to model weights file')
    parser.add_argument('--anchors', type=str, default='yolo_anchors.txt',
                        help='path to anchor definitions')
    parser.add_argument('--classes', type=str, default='face_classes.txt',
                        help='path to class definitions')
    parser.add_argument('--score', type=float, default=0.5,
                        help='the score threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='the iou threshold')
    parser.add_argument('--img-size', type=list, action='store',
                        default=(416, 416), help='input image size')
    args = parser.parse_args()
    return args

args = get_args()
data=input('Enter Image Name:')
photo=cv2.imread(data)
i=Image.open(data)
yolo=YOLO(args)
faces=yolo.detect_image(i)
print(faces,type(faces))
for y1,x1,y2,x2 in faces:
    sex,gender_prob=gender(photo,x1,y1,x2,y2)
    pro="He"if sex=="man" else "she"
    looks,prob=emotion(photo,x1,y1,x2,y2)
    print("Gender :",sex,"      ",gender_prob)
    print(pro,"looks",looks,"      ",prob)
    print("Mouth :",Detect_mouth_opening(photo[y1:y2,x1:x2],x1,y1,x2,y2))
    print("Glasses :",glasses(photo.copy(),x1,y1,x2,y2))
    cv2.imshow('output',photo[y1:y2,x1:x2])
    full=cv2.rectangle(photo,(x1,y1),(x2,y2),(255,255,255),1)
    cv2.waitKey(0)
cv2.imshow('Full',full)
cv2.waitKey(0)
