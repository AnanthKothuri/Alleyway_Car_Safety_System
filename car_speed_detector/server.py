# USAGE
# python server.py --conf config/config.json

# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector
from pyimagesearch.parseyolooutput import ParseYOLOOutput
from pyimagesearch.keyclipwriter import KeyClipWriter
from pyimagesearch.utils.conf import Conf
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import os
import time
import sys
import zmq

def overlap(rectA, rectB):
	# check if x1 of rectangle A is greater x2 of rectangle B or if
	# x2 of rectangle A is less than x1 of rectangle B, if so, then
	# both of them do not overlap and return False
	if rectA[0] > rectB[2] or rectA[2] < rectB[0]:
		return False

	# check if y1 of rectangle A is greater y2 of rectangle B or if
	# y2 of rectangle A is less than y1 of rectangle B, if so, then
	# both of them do not overlap and return False
	if rectA[1] > rectB[3] or rectA[3] < rectB[1]:
		return False

	# otherwise the two rectangles overlap and hence return True
	return True

# calculate the speed using k = 410
def distance(width):
	return (2782/width) - 2.74

def calculateSpeed(width1, width2, time):
	d1 = distance(width1)
	d2 = distance(width2)
	speed = (d1 -d2)/time # inches per second
	speedMPH = speed / (5280) * 3600 # miles per hour
	return round(speedMPH, 2)

def calculateTime(speed, width2):
	if speed ==0:
		return "literally never"
	d2 = distance(width2)
	return round(d2 / speed, 2)

SPEED_LIMIT = 30

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, 
	help="Path to the input configuration file")
ap.add_argument("-y", "--yolo", required=True,
	help="If we should utilize YOLO or just background subtraction")
args = vars(ap.parse_args())

# load the configuration file and initialize the ImageHub object
conf = Conf(args["conf"])

useYOLO = None
if args["yolo"] == "True":
	useYOLO = True
elif args["yolo"] == "False":
	useYOLO = False
	
imageHub = imagezmq.ImageHub()

# initialize the motion detector, the total number of frames read
# thus far, and the spatial dimensions of the frame
md = SingleMotionDetector(accumWeight=0.1)
total = 0
(W, H) = (None, None)

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([conf["yolo_path"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([conf["yolo_path"], "yolov3.weights"])
configPath = os.path.sep.join([conf["yolo_path"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the YOLO output parsing object
pyo = ParseYOLOOutput(conf)

# initialize key clip writer and the consecutive number of
# frames that have *not* contained any action
kcw = KeyClipWriter(bufSize=conf["buffer_size"])
consecFrames = 0
print("[INFO] starting advanced security surveillance...")

# initialize all variables for speed calc
firstWidth = None
firstTime = None

firstCheck = False
secondCheck = False

# start looping over all the frames
while True:
	# receive RPi name and frame from the RPi and acknowledge
	# the receipt
	(rpiName, frame) = imageHub.recv_image()
	imageHub.send_reply(b'OK')
	speed = 0
    
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# grab the current timestamp and draw it on the frame
	timestamp = datetime.now()
	cv2.putText(frame, timestamp.strftime(
		"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# if we do not already have the dimensions of the frame,
	# initialize it
	if H is None and W is None:
		(H, W) = frame.shape[:2]

	# if the total number of frames has reached a sufficient
	# number to construct a reasonable background model, then
	# continue to process the frame
	if total > conf["frame_count"]:
		# detect motion in the frame and set the update consecutive
		# frames flag as True
		motion = md.detect(gray)
		updateConsecFrames = True

		# if the motion object is not None, then motion has
		# occurred in the image
		if motion is not None:
			# set the update consecutive frame flag as false and
			# reset the number of consecutive frames with *no* action
			# to zero
			updateConsecFrames = False
			consecFrames = 0

			# uses input to decide whether to use only motion dection
            # or YOLO as well
			if useYOLO is False:
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),
							  (0,255,0), 2)

				if firstCheck is False:
					firstCheck = True
					firstTime = timestamp
					firstWidth = maxX-minX
				elif secondCheck is False:
					changeTime = (timestamp - firstTime).total_seconds()
					if changeTime > 0.3:
						speed = calculateSpeed(firstWidth, maxX-minX, changeTime)
						print("[INFO] the speed is {}".format(speed))
						estimatedTime = calculateTime(speed, maxX-minX)
						print("[INFO] ETA of object is {}".format(estimatedTime))
						print(" ")

						firstWidth = None
						firstTime = None

						firstCheck = False

			else:
				# construct a blob from the input frame and then perform
				# a forward pass of the YOLO object detector, giving us
				# our bounding boxes and associated probabilities
				blob = cv2.dnn.blobFromImage(frame, 1 / 255.0,
					(416, 416), swapRB=True, crop=False)
				net.setInput(blob)
				layerOutputs = net.forward(ln)

				# parse YOLOv3 output
				(boxes, confidences, classIDs) = pyo.parse(layerOutputs,
					LABELS, H, W)

				# apply non-maxima suppression to suppress weak,
				# overlapping bounding boxes
				idxs = cv2.dnn.NMSBoxes(boxes, confidences,
					conf["confidence"], conf["threshold"])

				# ensure at least one detection exists
				if len(idxs) > 0:
					# loop over the indexes we are keeping
					for i in idxs.flatten():
						# extract the bounding box coordinates
						(x, y) = (boxes[i][0], boxes[i][1])
						(w, h) = (boxes[i][2], boxes[i][3])

						# loop over all the unauthorized zones
						for zone in conf["unauthorized_zone"]:
							# store the coordinates of the detected
							# object in (x1, y1, x2, y2) format
							obj = (x, y, x + w, y + h)

							# check if there is NOT a overlap between the
							# object and the zone, if so, then skip this
							# iteration
							if not overlap(obj, zone):
								continue

							# otherwise there is overlap between the
							# object and the zone
							else:
								# draw a bounding box rectangle and label on the
								# frame
								color = [int(c) for c in COLORS[classIDs[i]]]
								cv2.rectangle(frame, (x, y), (x + w, y + h),
									color, 2)
								text = "{}: {:.4f}".format(LABELS[classIDs[i]],
									confidences[i])
								y = (y - 15) if (y - 15) > 0 else h - 15
								cv2.putText(frame, text, (x, y),
									cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

								# used to obtain the pixel width of the object during testing
								# print("[INFO] width of box is {}".format(w))

								# utilize speed logic between two frames
								if firstCheck is False:
									firstCheck = True
									firstTime = timestamp
									firstWidth = w
								elif secondCheck is False:
									changeTime = (timestamp - firstTime).total_seconds()
									if changeTime > 0.3:
										speed = calculateSpeed(firstWidth, w, changeTime)
										print("[INFO] the speed is {}".format(speed))
										estimatedTime = calculateTime(speed, w)
										print("[INFO] ETA of object is {}".format(estimatedTime))
										print(" ")

										firstWidth = None
										firstTime = None

										firstCheck = False

								# break out of the loop since the object
								# overlaps with at least one zone
								break

			# if we are not already recording, start recording
			if not kcw.recording and speed > SPEED_LIMIT + 10:
				# store the day's date and check if output directory
				# exists, if not, then create
				date = timestamp.strftime("%Y-%m-%d")
				os.makedirs(os.path.join(conf["output_path"], date),
							exist_ok=True)

				# build the output video path and start recording
				p = "{}/{}/{}.avi".format(conf["output_path"], date,
										  timestamp.strftime("%H%M%S"))
				kcw.update(frame)
				kcw.start(p, cv2.VideoWriter_fourcc(*conf["codec"]),
						  conf["fps"])

		# loop over the unauthorized zones
		for (x1, y1, x2, y2) in conf["unauthorized_zone"]:
			# draw the zone on the frame
			cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

		# otherwise, no action has taken place in this frame, so
		# increment the number of consecutive frames that contain
		# no action
		if updateConsecFrames:
			consecFrames += 1

		# update the key frame clip buffer
		kcw.update(frame)

		# if we are recording and reached a threshold on consecutive
		# number of frames with no action, stop recording the clip
		if kcw.recording and consecFrames == conf["buffer_size"]:
			kcw.finish()

	# update the background model and increment the total number
	# of frames read thus far
	md.update(gray)
	total += 1

	# show the frame
	cv2.imshow("{}".format(rpiName), frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		print("[INFO] Quitting Video Stream ...")
		break

# if we are in the middle of recording a clip, wrap it up
if kcw.recording:
	kcw.finish()

# do a bit of cleanup
cv2.destroyAllWindows()
