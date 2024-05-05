# USAGE
# python read_frames_slow.py --video videos/jurassic_park_intro.mp4

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

# from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

import os



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



# ---------------------------------------------------------------------------------------
# set arguments
# ---------------------------------------------------------------------------------------

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True,
# 	help="path to input video file")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20170206_file-video-stream"



# ---------------------------------------------------------------------------------------
# read frames by normal (slow) method
# ---------------------------------------------------------------------------------------

video_file = os.path.join(base_path, "videos\\jurassic_park_intro.mp4")


# open a pointer to the video stream and start the FPS timer
stream = cv2.VideoCapture(video_file)


fps = FPS().start()


# ----------
# loop over frames from the video file stream

while True:
	# grab the frame from the threaded video file stream
	(grabbed, frame) = stream.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and convert it to grayscale (while still
	# retaining 3 channels)
	frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame, frame, frame])

	# display a piece of text to the frame (so we can benchmark
	# fairly against the fast method)
	cv2.putText(frame, "Slow Method", (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	

	# show the frame and update the FPS counter
	cv2.imshow("Frame", frame)
	cv2.waitKey(1)
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
