
import numpy as np
import cv2
import imutils

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage


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


def img_proc(img):
    kernel = np.ones((5, 5), np.uint8)
    # img_obj = adjust_gamma(img, gamma=1.75)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_obj = cv2.GaussianBlur(img_obj, (7, 7), 0)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj


def anime_filter1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    edge = cv2.blur(gray, (3, 3))
    edge = cv2.Canny(edge, 50, 150, apertureSize=3)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    # by pyrMeanShiftFiltering
    img = cv2.pyrMeanShiftFiltering(img, 5, 20)
    return cv2.subtract(img, edge)


def sub_color(src, K):
    Z = src.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((src.shape))


def anime_filter2(img, K):
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    edge = cv2.blur(gray, (3, 3))
    edge = cv2.Canny(edge, 50, 150, apertureSize=3)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    # by k-means
    img = sub_color(img, K)
    return cv2.subtract(img, edge)


# ------------------------------------------------------------------------------------------------------
# extract image
# ------------------------------------------------------------------------------------------------------

# video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly'
#
# # ----------
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


# rate = 4
rate = 1


# ----------
# imgA
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
    cv2.imshow('image', frame2)
    print(j)
    if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    j += 1

cv2.destroyAllWindows()


# ----------
# imgB
# vid2 = cv2.VideoCapture(video_file1)

vid2 = cv2.VideoCapture(video_file4)
# start_frame2 = 3100
# end_frame2 = start_frame2 + 1

start_frame2 = 2050
end_frame2 = start_frame2 + 1

vid2.set(1, start_frame2)

j = start_frame2

while j <= end_frame2:
    is_read, frame = vid2.read()
    if not (is_read):
        break
    frame3 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
    cv2.imshow('image', frame3)
    print(j)
    if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    j += 1

cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------------------------
# preprocess
# ------------------------------------------------------------------------------------------------------

img = frame2
# img = frame3

# xmin = 100
# xmax = 300
# ymin = 0
# ymax = 75
# img = frame2[ymin:ymax, xmin:xmax]


# perform pyramid mean shift filtering to aid the thresholding step
shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
# shifted = cv2.pyrMeanShiftFiltering(img, 11, 21)
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)


# ------------------------------------------------------------------------------------------------------
# find countours
# ------------------------------------------------------------------------------------------------------

# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
# print(len(cnts))
#
#
# # ----------
# # display contours
#
# clone = img.copy()
#
# for (i, c) in enumerate(cnts):
#     ((x, y), _) = cv2.minEnclosingCircle(c)
#     cv2.putText(clone, "#{}".format(i+1), (int(x) - 10, int(y)),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#     cv2.drawContours(clone, [c], -1, (0, 255, 0), 2)
#
#
# cv2.imshow("03 Image+Contours", clone)
# cv2.imshow("02 Thresholded", thresh)
# cv2.imshow("01 Shifted", shifted)
# cv2.waitKey(0)
#


# ------------------------------------------------------------------------------------------------------
# sure background
# ------------------------------------------------------------------------------------------------------

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

sure_bg = cv2.dilate(thresh, kernel, iterations=1)


# ------------------------------------------------------------------------------------------------------
# find local peak of distance
# ------------------------------------------------------------------------------------------------------

# compute the exact Euclidean distance from every binary pixel to the nearest zero pixel,
# then find peaks in this distance map

D = ndimage.distance_transform_edt(thresh)

localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
# localMax = peak_local_max(D, indices=False, min_distance=40, labels=thresh)


# ----------
print(np.min(D))
print(np.max(D))

D0 = ((np.max(D) - D) / np.max(D) * 255).astype('uint8')


# ------------------------------------------------------------------------------------------------------
# connected component analysis
# ------------------------------------------------------------------------------------------------------

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then apply the Watershed algorithm

markers = ndimage.label(localMax, structure=np.ones((3,3)))[0]

labels = watershed(-D, markers, mask=thresh)

print(markers)
print(labels)

print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))


# ------------------------------------------------------------------------------------------------------
# segmentation by watershed
# ------------------------------------------------------------------------------------------------------

clone1 = img.copy()
clone2 = img.copy()

# ----------
for label in np.unique(labels):
    if label == 0:
        continue

    mask = np.zeros(gray.shape, dtype='uint8')
    mask[labels == label] = 255

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)

    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(clone1, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(clone1, "#{}".format(label), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# ----------
label_lst = np.unique(labels)
tmp = []
# ignore 0:background 1:boundary
for label in label_lst[2:]:
    target = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    tmp.append(contours[0])
cv2.drawContours(clone2, tmp, -1, color=(0, 0, 255), thickness=2)


# ------------------------------------------------------------------------------------------------------
# anime filter
# ------------------------------------------------------------------------------------------------------

anime1 = anime_filter1(img)
anime2 = anime_filter2(img, 10)


cv2.imshow("08 anime filter2: kmeans", anime2)
cv2.imshow("07 anime filter1: mean shift filtering", anime1)
cv2.imshow("06 Image+Contours", clone2)
cv2.imshow("05 Sure Background", sure_bg)
cv2.imshow("04 Labels", clone1)
cv2.imshow("03 Distance", D0)
cv2.imshow("02 Thresholded", thresh)
cv2.imshow("01 Shifted", shifted)
cv2.waitKey(0)


# ----------
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(labels, cmap="tab20b")
plt.show()


cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------------------------
# detecting in video frame
# ------------------------------------------------------------------------------------------------------

video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(video_orig_path, vid_file)

rate = 1
fps = 30

# xmin = 100
# xmax = 300
# ymin = 0
# ymax = 75

xmin = 100
xmax = 300 + 60
ymin = 0
ymax = 75 + 25

# ----------
vid = cv2.VideoCapture(video_file)
start_frame = 6200
end_frame = 3600 * 30

vid.set(1, start_frame)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

lst = []
j = start_frame - 1
st = time.time()

while j <= end_frame:

    is_read, frame = vid.read()
    if not (is_read):
        break

    frame = frame[ymin:ymax, xmin:xmax]

    j += 1

    shifted = cv2.pyrMeanShiftFiltering(frame, 21, 51)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=30, labels=thresh)
    D0 = ((np.max(D) - D) / np.max(D) * 255).astype('uint8')
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    clone1 = frame.copy()
    # clone2 = frame.copy()

    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(gray.shape, dtype='uint8')
        mask[labels == label] = 255
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(clone1, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(clone1, "#{}".format(label), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # label_lst = np.unique(labels)
    # tmp = []
    # # ignore 0:background 1:boundary
    # for label in label_lst[2:]:
    #     target = np.where(labels == label, 255, 0).astype(np.uint8)
    #     contours, hierarchy = cv2.findContours(
    #         target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    #     )
    #     tmp.append(contours[0])
    # cv2.drawContours(clone2, tmp, -1, color=(0, 0, 255), thickness=2)

    cv2.imshow("04 Labels", clone1)
    # cv2.imshow("04 Labels2", clone2)

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

ed = time.time()
print(f'takes {ed - st: .2f} secs')

cv2.waitKey(0)

cv2.destroyAllWindows()


