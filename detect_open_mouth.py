from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

# construct the argument parse and parse the arguments


# start the video stream thread

# loop over frames from the video stream
def Detect_mouth_opening(image,x1,y1,x2,y2):
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
	help="path to facial landmark predictor")
	ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
	args = vars(ap.parse_args())

# define one constants, for mouth aspect ratio to indicate open mouth
	MOUTH_AR_THRESH = 0.75

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
	predictor = dlib.shape_predictor(args["shape_predictor"])
	(mStart, mEnd) = (49, 68)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rect=dlib.rectangle(x1,y1,x2,y2)
	shape = predictor(gray,rect)
	shape = face_utils.shape_to_np(shape)
	mouth = shape[mStart:mEnd]

	mouthMAR = mouth_aspect_ratio(mouth)
	mar = mouthMAR
        # Draw text if mouth is open
	if mar > MOUTH_AR_THRESH:
			return 'Open'
	else:
			return'closed'
