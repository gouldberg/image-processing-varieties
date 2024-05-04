
import numpy as np
import cv2
import imutils

from imutils.video import VideoStream
import time


base_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis'


def img_proc(img):
    kernel = np.ones((3, 3), np.uint8)
    # img_obj = adjust_gamma(img, gamma=0.8)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (3, 3), 0)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj

# -----------------------------------------------------------------------------------------------
# objective image
# -----------------------------------------------------------------------------------------------

# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303165000_10min.mp4'
# start_frame = 2000
# end_frame = start_frame + 300

# start_frame = 5000
# end_frame = start_frame + 10

# something on work station
# start_frame = 2800
# end_frame = start_frame + 200

# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210223133400_10min.mp4'
# start_frame = 1
# end_frame = start_frame + 10
# start_frame = 1000
# end_frame = start_frame + 100

# vid_file = 'Gamma_Assembly_192_168_32_69_1_20201130065000_10min.mp4'
# start_frame = 8000
# end_frame = start_frame + 100


video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(video_orig_path, vid_file)

start_frame = 1

vid = cv2.VideoCapture(video_file)
vid.set(1, start_frame)

rate = 1

j = start_frame
end_frame = 3600*30

while j <= end_frame:
    is_read, frame = vid.read()
    if not (is_read):
        break
    frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
    # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
    cv2.imshow('image', frame2)
    print(j)
    if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    j += 1

cv2.destroyAllWindows()

image = frame2


# -----------------------------------------------------------------------------------------------
# static saliency
# This static saliency detector operates on the log-spectrum of an image,
# computes saliency residuals in this spectrum,
# and then maps the corresponding salient locations back to the spatial domain
#
#   - 1. spectral residual detector
# -----------------------------------------------------------------------------------------------

saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

(success, saliencyMap) = saliency.computeSaliency(image)

saliencyMap = (saliencyMap * 255).astype("uint8")

cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.waitKey(0)

cv2.destroyAllWindows()


# -----------------------------------------------------------------------------------------------
# static saliency
#   - 2. static fine grained saliency detector
# -----------------------------------------------------------------------------------------------

saliency = cv2.saliency.StaticSaliencyFineGrained_create()

(success, saliencyMap) = saliency.computeSaliency(image)


# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map

threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# show the images
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
cv2.imshow("Thresh", threshMap)
cv2.waitKey(0)


# -----------------------------------------------------------------------------------------------
# objectness saliency
# -----------------------------------------------------------------------------------------------

model_path = 'C:\\Users\\kosei-wada\\Desktop\\reference\\20180716_saliency-detection\\objectness_trained_model'

xmin = 80
xmax = 280
ymin = 0
ymax = 140

# ----------
saliency = cv2.saliency.ObjectnessBING_create()

saliency.setTrainingPath(model_path)

frame = image[ymin:ymax, xmin:xmax]
(success, saliencyMap) = saliency.computeSaliency(frame)

numDetections = saliencyMap.shape[0]

max_detections = 10

for i in range(0, min(numDetections, max_detections)):

    (startX, startY, endX, endY) = saliencyMap[i].flatten()

    # randomly generate a color for the object and draw it on the image
    output = frame.copy()
    color = np.random.randint(0, 255, size=(3,))
    color = [int(c) for c in color]
    cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)


# -----------------------------------------------------------------------------------------------
# motion saliency
# -----------------------------------------------------------------------------------------------

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')

    return cv2.LUT(image, table)

def img_proc(img):
    kernel = np.ones((3, 3), np.uint8)
    img_obj = adjust_gamma(img, gamma2)
    # img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_obj = cv2.GaussianBlur(img_obj, (3, 3), 0)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    # img_obj = cv2.cvtColor(img_obj, cv2.COLOR_GRAY2BGR)
    return img_obj


video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(video_orig_path, vid_file)

# video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video'
# vid_file = '20210420_063007_60min.mp4'
# video_file = os.path.join(video_orig_path, vid_file)


xmin = 80
xmax = 280
ymin = 0
ymax = 140


vs = cv2.VideoCapture(video_file)

start_frame = (28*60+50)*30
vs.set(1, start_frame)

is_read, frame = vs.read()
# frame = imutils.resize(frame, width=160)

# 0. spectral residual
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

# 1.
# saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
# saliency.setImagesize(frame.shape[1], frame.shape[0])
# saliency.init()

# 2.
# saliency = cv2.saliency.StaticSaliencyFineGrained_create()

# 3.
# model_path = 'C:\\Users\\kosei-wada\\Desktop\\reference\\20180716_saliency-detection\\objectness_trained_model'
# saliency = cv2.saliency.ObjectnessBING_create()
# saliency.setTrainingPath(model_path)


time.sleep(2.0)

st = time.time()
while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    is_read, frame = vs.read()
    if not is_read:
        break

    # is_read, frame = vs.read()
    # if not is_read:
    #     break

    # frame = imutils.resize(frame, width=160)

    # frame = img_proc(frame)
    # frame = frame[ymin:ymax, xmin:xmax]

    # if our saliency object is None, we need to instantiate it
    # if saliency is None:
    #     saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
    #     saliency.setImagesize(frame.shape[1], frame.shape[0])
    #     saliency.init()
    #

    # ----------
    # 1. Motion Saliency Bin Wang Apr2014
    # convert the input frame to grayscale and compute the saliency
    # map based on the motion model
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # (success, saliencyMap) = saliency.computeSaliency(gray)
    # saliencyMap = (saliencyMap * 255).astype("uint8")

    # ----------
    (success, saliencyMap) = saliency.computeSaliency(frame)


    # display the image to our screen
    cv2.imshow("Frame", frame)
    cv2.imshow("Map", saliencyMap)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

ed = time.time()
print(f'{ed - st: .2f}')

cv2.destroyAllWindows()
vs.release()



