# USAGE
# python frame_counter.py --video videos/jurassic_park_trailer.mp4

sys.path.append("C:\\Users\\kouse\\kw\\venv\\Lib\\site-packages")

# from imutils.video import count_frames
import argparse
import cv2

import os



# ------------------------------------------------------------------------------------------
# get_opencv_major_version(), is_cv3(), count_frames(), count_frames_manual()
# ------------------------------------------------------------------------------------------

def get_opencv_major_version(lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib

    # return the major version number
    return int(lib.__version__.split(".")[0])


def is_cv3(or_better=False):
	# grab the OpenCV major version number
	major = get_opencv_major_version()

	# check to see if we are using *at least* OpenCV 3
	if or_better:
		return major >= 3

	# otherwise we want to check for *strictly* OpenCV 3
	return major == 3


def count_frames(path, override=False):
	# grab a pointer to the video file and initialize the total
	# number of frames read
	video = cv2.VideoCapture(path)
	total = 0

	# if the override flag is passed in, revert to the manual
	# method of counting frames
	if override:
		total = count_frames_manual(video)

	# otherwise, let's try the fast way first
	else:
		# lets try to determine the number of frames in a video
		# via video properties; this method can be very buggy
		# and might throw an error based on your OpenCV version
		# or may fail entirely based on your which video codecs
		# you have installed
		try:
			# check if we are using OpenCV 3
			if is_cv3():
				total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

			# otherwise, we are using OpenCV 2.4
			else:
				total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

		# uh-oh, we got an error -- revert to counting manually
		except:
			total = count_frames_manual(video)

	# release the video file pointer
	video.release()

	# return the total number of frames in the video
	return total


def count_frames_manual(video):
	# initialize the total number of frames read
	total = 0

	# loop over the frames of the video
	while True:
		# grab the current frame
		(grabbed, frame) = video.read()

		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break

		# increment the total number of frames read
		total += 1

	# return the total number of frames in the video file
	return total



# ------------------------------------------------------------------------------------------
# set arguments
# ------------------------------------------------------------------------------------------

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", required=True,
# 	help="path to input video file")
# ap.add_argument("-o", "--override", type=int, default=-1,
# 	help="whether to force manual frame count")
# args = vars(ap.parse_args())


base_path = "C:\\Users\\kouse\\Desktop\\ImageProcessing\\PyImageSearchPlus\\20170109_count-frames-opencv"



# ------------------------------------------------------------------------------------------
# count total number of frames in the video file
# ------------------------------------------------------------------------------------------

video_file = os.path.join(base_path, "videos\\jurassic_park_trailer.mp4")


# ----------
# count the total number of frames in the video file

# Method #1: The fast, efficient way using the built-in properties OpenCV provides us
# to access the video file meta information and return the total number of frames.

# Method #2: The slow, inefficient technique that requires us to manually loop over
# each frame and increment a counter for each frame we’ve read.

# The problem here is that Method #1 is buggy as all hell based on your OpenCV version
# and video codecs installed.

# Method #1
total0 = count_frames(video_file, override=False)

# Method #2
total1 = count_frames(video_file, override=True)


# ----------
print("[INFO] Method 1:   {:,} total frames read from {}".format(total0,
	video_file[video_file.rfind(os.path.sep) + 1:]))


print("[INFO] Method 2:   {:,} total frames read from {}".format(total1,
	video_file[video_file.rfind(os.path.sep) + 1:]))

