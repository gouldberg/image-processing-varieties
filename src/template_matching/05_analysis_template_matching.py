
import numpy as np
import cv2
import imutils

from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create
from imutils.object_detection import non_max_suppression

import itertools

import time

from tqdm import tqdm
import matplotlib.pyplot as plt

import csaps


# ------------------------------------------------------------------------------------------------------
# support functions
# ------------------------------------------------------------------------------------------------------

def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted


def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
	return rotated


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(max(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')

    return cv2.LUT(image, table)


def img_proc_tmplt(img):
    kernel = np.ones((5, 5), np.uint8)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (7, 7), 0)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj

def img_proc_orig(img):
    kernel = np.ones((5, 5), np.uint8)
    img_obj = adjust_gamma(img, gamma=1)
    img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (7, 7), 0)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj


# ------------------------------------------------------------------------------------------------------
# check video
# ------------------------------------------------------------------------------------------------------

# video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly'

# # non-champion
# vid_file1 = 'Gamma_Assembly_192_168_32_69_1_20201130110000_10min.mp4'
# video_file1 = os.path.join(video_orig_path, vid_file1)
#
# vid_file2 = 'Gamma_Assembly_192_168_32_69_1_20201130180000_10min.mp4'
# video_file2 = os.path.join(video_orig_path, vid_file2)
#
# # champion
# vid_file3 = 'Gamma_Assembly_192_168_32_69_1_20210223133400_10min.mp4'
# video_file3 = os.path.join(video_orig_path, vid_file3)
#
# vid_file4 = 'Gamma_Assembly_192_168_32_69_1_20210303161000_10min.mp4'
# video_file4 = os.path.join(video_orig_path, vid_file4)

video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(video_orig_path, vid_file)

rate = 1


# ----------
vid = cv2.VideoCapture(video_file)
start_frame = 1
end_frame = start_frame + 10

vid.set(1, start_frame)

j = start_frame

while j <= end_frame:
    is_read, frame = vid.read()
    if not (is_read):
        break
    frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
    frame2 = img_proc_orig(frame2)
    cv2.imshow('image', frame2)
    print(j)
    if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    j += 1

cv2.destroyAllWindows()


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

vid = cv2.VideoCapture(video_file)
vid.set(1, start_frame)

j = start_frame

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

cv2.imwrite(os.path.join(base_path, '01_data\\image_template\\source.png'), frame2)

image = frame2


# -----------------------------------------------------------------------------------------------
# load template image
# -----------------------------------------------------------------------------------------------

# note that those are gamma corrected by 1.75
# tmplt_work_station = os.path.join(base_path, '01_data\\image_template\\work_station.png')
# tmplt_work_station2 = os.path.join(base_path, '01_data\\image_template\\work_station_2.png')
# tmplt_parts_box_0 = os.path.join(base_path, '01_data\\image_template\\parts_box_0.png')
# tmplt_parts_box_1 = os.path.join(base_path, '01_data\\image_template\\parts_box_1.png')
# tmplt_parts_box_2 = os.path.join(base_path, '01_data\\image_template\\parts_box_2.png')
# tmplt_screw_box = os.path.join(base_path, '01_data\\image_template\\screw_box.png')

tmplt_screwdriver_keypoint = os.path.join(base_path, '01_data\\image_template\\screwdriver_keypoint.png')

template = cv2.imread(tmplt_screwdriver_keypoint)

# size is fixed...
(tH, tW) = template.shape[:2]


# ----------
def img_proc_tmplt(img):
    kernel = np.ones((3, 3), np.uint8)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (3, 3), 0)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj

def img_proc_orig(img):
    kernel = np.ones((3, 3), np.uint8)
    img_obj = adjust_gamma(img, gamma=1)
    img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (3, 3), 0)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj

imageGray = img_proc_orig(image)
templateGray = img_proc_tmplt(template)

cv2.imshow("Image", imageGray)
cv2.imshow("Template", templateGray)
cv2.waitKey(0)


# -----------------------------------------------------------------------------------------------
# perform template matching
# -----------------------------------------------------------------------------------------------

# values in the range[0.8, 0.95] typically work the best
# --> but here should be less than 0.75
# thresh = 0.72
# thresh = 0.8
thresh = 0.5

print("[INFO] performing template matching...")
result = cv2.matchTemplate(imageGray, templateGray,
	cv2.TM_CCOEFF_NORMED)

print(len(result))


# find all locations in the result map where the matched value is
# greater than the threshold, then clone our original image so we
# can draw on it
(yCoords, xCoords) = np.where(result >= thresh)
clone = image.copy()
print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))


# ----------
# loop over our starting (x, y)-coordinates
# for (x, y) in zip(xCoords, yCoords):
# 	# draw the bounding box on the image
# 	cv2.rectangle(clone, (x, y), (x + tW, y + tH),
# 		(255, 0, 0), 3)
#
# # show our output image *before* applying non-maxima suppression
# cv2.imshow("Before NMS", clone)
# cv2.waitKey(0)

# initialize our list of rectangles
rects = []

# loop over the starting (x, y)-coordinates again
for (x, y) in zip(xCoords, yCoords):
	# update our list of rectangles
	rects.append((x, y, x + tW, y + tH))

# apply non-maxima suppression to the rectangles
pick = non_max_suppression(np.array(rects))
print("[INFO] {} matched locations *after* NMS".format(len(pick)))

# loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
	# draw the bounding box on the image
	cv2.rectangle(clone, (startX, startY), (endX, endY),
		(255, 0, 0), 3)

# show the output image
cv2.imshow("After NMS", clone)
cv2.waitKey(0)


# -----------------------------------------------------------------------------------------------
# loop over the images to find the template in
# -----------------------------------------------------------------------------------------------

# template matching is not ideal if you are trying to match rotated objects or
# objects that exhibit non-affine transformations.
# If you are concerned with these types of transformations
# you are better of jumping right to keypoint matching.

video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(video_orig_path, vid_file)

rate = 1
fps = 30

xmin = 100
xmax = 300
ymin = 0
ymax = 75

# ----------
vid = cv2.VideoCapture(video_file)
start_frame = 1
end_frame = 3600 * 30

vid.set(1, start_frame)

j = start_frame - 1

v_flag = False

lst = []

st = time.time()

while j <= end_frame:
    is_read, frame = vid.read()
    if not (is_read):
        break
    is_read, frame = vid.read()
    if not (is_read):
        break

    j += 2

    frame = frame[ymin:ymax, xmin:xmax]
    imageGray = img_proc_orig(frame)

    found = None

    # loop over the scales of the image
    # for scale in np.linspace(0.05, 2.0, 200)[::-1]:
    # for scale, angle in itertools.product(np.linspace(0.05, 2.0, 81)[::-1], np.linspace(0, 360, 81)[::1]):

    # for scale, angle in itertools.product(np.linspace(0.5, 2.0, 3)[::-1], np.linspace(0, 360, 4)[::1]):

    scale = 1.0
    angle = 0.0

    # -----------------------------------------
    # print(f'{scale} - {angle}')
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    # resized = resize(imageGray, width = int(imageGray.shape[1] * scale))
    # resized = rotate(imageGray, angle=angle, center=None, scale=scale)
    # cv2.imshow('resized', resized)
    # cv2.waitKey(0)

    r = 1.0
    # r = imageGray.shape[1] / float(resized.shape[1])

    # if the resized image is smaller than the template, then break from the loop
    # if resized.shape[0] < tH or resized.shape[1] < tW:
    #     break

    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    # edged = cv2.Canny(resized, 50, 200)

    # edged = resized
    # result = cv2.matchTemplate(resized, templateGray, cv2.TM_CCOEFF_NORMED)
    result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)

    # takes correlation result and returns a 4-tuple which includes the minimum correlation value,
    # the (x,y)-coordinate of the minimum value, and the (x,y)-coordinate of the maximum value, respectively.
    # We are only interested in the maximum value and (x,y)-coordinate so we keep the maximums and discard the minimums
    (, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    # check to see if the iteration should be visualized
    # if v_flag == True:
    #     # draw a bounding box around the detected region
    #     clone = np.dstack([edged, edged, edged])
    #     cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
    #         (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
    #     cv2.imshow("Visualize", clone)
    #     cv2.waitKey(0)

    # if we have found a new maximum correlation value, then update
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r, angle, scale)

    # -----------------------------------------
    # print(found)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r, angle, scale) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # draw a bounding box around the detected result and display the image
    clone2 = frame.copy()
    # clone2 = rotate(clone2, angle=angle, scale=1.0)
    cv2.rectangle(clone2, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", clone2)

    print((maxLoc[0] + tW/2, maxLoc[1] + tH/2))
    lst.append([(j-1)/fps, maxLoc[0], maxLoc[1], maxLoc[0] + tW, maxLoc[1] + tH, maxLoc[0] + tW/2, maxLoc[1] + tH/2])

    # if cv2.waitKey(3) & 0xFF == ord('q'):
    #     break

ed = time.time()
print(f'takes {ed - st: .2f} secs')

cv2.waitKey(0)

cv2.destroyAllWindows()


# ----------
traj = pd.DataFrame(lst)

traj.columns = ['sec_orig', 'startX', 'startY', 'endX', 'endY', 'xcnt', 'ycnt']

traj.xcnt.plot()
traj.ycnt.plot()

traj.to_csv(os.path.join(base_path, '03_datamart\\tmplt_tracking.csv'), index=False)


# -----------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------

def plot_spectrogram(fs, s,  Lf, noverlap=None):
    S = STFT(s, Lf, noverlap)

    S = S + 1e-18
    P = 20 * np.log10(np.abs(S))
    P = P - np.max(P) # normalization
    vmin = -150
    if np.min(P) > vmin:
        vmin = np.min(P)
    m = np.linspace(0, s.shape[0]/fs, num=P.shape[1])
    k = np.linspace(0, fs/2, num=P.shape[0])
    plt.figure()
    plt.pcolormesh(m, k, P, cmap = 'jet', vmin=-150, vmax=0)
    plt.title("Spectrogram")
    plt.xlabel("time")
    plt.ylabel("frequency[Hz]")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def STFT(s, Lf, noverlap=None):
    if noverlap==None:
        noverlap = Lf//2
    l = s.shape[0]
    win = np.hanning(Lf)
    Mf = Lf//2 + 1
    Nf = int(np.ceil((l-noverlap)/(Lf-noverlap)))-1
    S = np.empty([Mf, Nf], dtype=np.complex128)
    for n in tqdm(range(Nf)):
        S[:,n] = np.fft.rfft(s[(Lf-noverlap)*n:(Lf-noverlap)*n+Lf] * win, n=Lf, axis=0)
    return S


# ----------
# read data

traj = pd.read_csv(os.path.join(base_path, '03_datamart\\tmplt_tracking.csv'))


# ----------
# length of frame(window)
# Lf = 2**8

# # noverlap = Lf//2
# noverlap = None
# plot_spectrogram(fs=15, s=traj.xcnt[0:500*15], Lf=Lf, noverlap=noverlap)
# traj.xcnt[(100*15):(150*15)].plot()


# ----------
Lf = 2**6

fs = 15

noverlap = Lf//2

s = traj.xcnt
s_sp = csaps.csaps(list(range(len(s))), s, list(range(len(s))), smooth=0.005)
s_res = s - s_sp

win = np.hanning(Lf)

Mf = Lf // 2 + 1

Nf = int(np.ceil((len(s_res) - noverlap) / (Lf - noverlap))) - 1

print(Lf / fs)
print(len(s))
print(len(win))
print(Mf)
print(Nf)


S = np.empty([Mf, Nf], dtype=np.complex128)

for n in range(Nf):
    st = (Lf - noverlap) * n
    ed = (Lf - noverlap) * n + Lf
    s_res[st:ed]
    S[:, n] = np.fft.rfft(s_res[(Lf - noverlap) * n:(Lf - noverlap) * n + Lf] * win, n=Lf, axis=0)

S = S.T


# ----------
S[0]
len(S[0])

# P = 20 * np.log10(np.abs(S))
# P = P - np.max(P)  # normalization

P = np.abs(S)

P[0]

len(P)

sum(P[0]) - P[0][0]

for i in list(range(1500,1520,1)):
    pd.DataFrame(P[i]).plot()




# m = np.linspace(0, s.shape[0] / fs, num=P.shape[1])
# k = np.linspace(0, fs / 2, num=P.shape[0])





