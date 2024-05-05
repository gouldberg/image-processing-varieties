# USAGE
# python fps_demo.py
# python fps_demo.py --display 1

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

# from __future__ import print_function
# from imutils.video import WebcamVideoStream
# from imutils.video import FPS
import argparse
import imutils
import cv2

import os

from threading import Thread
import cv2



# ----------------------------------------------------------------------------------------
# class WebcomVideoStream
# ----------------------------------------------------------------------------------------

class WebcamVideoStream:
	def __init__(self, src=0, name="WebcamVideoStream"):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the thread name
		self.name = name

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, name=self.name, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


# ----------------------------------------------------------------------------------------
# class FPS
# ----------------------------------------------------------------------------------------

import datetime

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()



# ----------------------------------------------------------------------------------------
# set arguments
# ----------------------------------------------------------------------------------------

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-n", "--num-frames", type=int, default=100,
# 	help="# of frames to loop over for FPS test")
# ap.add_argument("-d", "--display", type=int, default=-1,
# 	help="Whether or not frames should be displayed")
# args = vars(ap.parse_args())



# ----------------------------------------------------------------------------------------
# normal stream
# ----------------------------------------------------------------------------------------

# grab a pointer to the video stream and initialize the FPS counter
print("[INFO] sampling frames from webcam...")

stream = cv2.VideoCapture(0)

fps = FPS().start()



# ----------
num_frames = 100

display_flag = -1


while fps._numFrames < num_frames:
	# grab the frame from the stream and resize it to have a maximum
	# width of 400 pixels
	(grabbed, frame) = stream.read()
	frame = imutils.resize(frame, width=400)

	# check to see if the frame should be displayed to our screen
	if display_flag > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

	# update the FPS counter
	fps.update()


# ----------
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))



# ----------
# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()



# ----------------------------------------------------------------------------------------
# Threaded video stream
# ----------------------------------------------------------------------------------------

# created a *threaded *video stream, allow the camera senor to warmup,
# and start the FPS counter

print("[INFO] sampling THREADED frames from webcam...")

vs = WebcamVideoStream(src=0).start()

fps = FPS().start()


# ----------
num_frames = 100

display_flag = -1


# loop over some frames...this time using the threaded stream

while fps._numFrames < num_frames:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# check to see if the frame should be displayed to our screen
	if display_flag > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

	# update the FPS counter
	fps.update()



# ----------
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))



# ----------
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

