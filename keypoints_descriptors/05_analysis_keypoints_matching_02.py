
import numpy as np
import cv2
import imutils

from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create


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


# ------------------------------------------------------------------------------------------------------
# get reference image
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


# ----------
# 0419 - 0421
vid_original_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video\\originals'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(vid_original_path, vid_file)

rate = 1

# ----------
# reference image

vid = cv2.VideoCapture(video_file)
fps = 30

screw_time_list = [(220,240),(1655,1675),(1750,1770),(1840,1860)]
screw_time_list = [(screw_time_list[i][0] * fps, screw_time_list[i][1] * fps) for i in range(len(screw_time_list))]

# for i in range(len(screw_time_list)):
for i in [0]:
    start_frame = screw_time_list[i][0]
    end_frame = screw_time_list[i][1]

    vid.set(1, start_frame)

    j = start_frame

    while j <= end_frame:
        is_read, frame = vid.read()
        if not (is_read):
            break
        frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
        # only screwing image
        frame3 = frame2[10:120, 90:320]
        # frame3 = img_proc(frame3)
        cv2.imshow('image', frame3)
        print(j)
        if cv2.waitKey(3) & 0xFF == ord('q'):
                break
        j += 1

cv2.destroyAllWindows()


# ----------
# reference image
train_img = frame3


# ------------------------------------------------------------------------------------------------------
# keypoints in reference image
# ------------------------------------------------------------------------------------------------------

def img_proc(img):
    kernel = np.ones((3, 3), np.uint8)
    # img_obj = adjust_gamma(img, gamma=1.75)
    # img_obj = cv2.pyrMeanShiftFiltering(img, 11, 21)
    # img_obj = cv2.pyrMeanShiftFiltering(img, 5, 11)
    img_obj = cv2.pyrMeanShiftFiltering(img, 15, 15)
    img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

    # img_obj = cv2.GaussianBlur(img_obj, (5, 5), 0)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    # img_obj = auto_canny(img_obj)
    return img_obj

cv2.imshow('image', img_proc(train_img))


# detector
# dct = 'FAST'
# dct = 'ORB'
# dct = 'BRISK'
dct = 'SIFT'
# dct = 'DOG'
# dct = 'FAST'
# dct = 'FASTHESSIAN'
# dct = 'SURF'
# dct = 'GFTT'
# dct = 'HARRIS'
# dct = 'MSER'
# dct = 'STAR'

# dct = 'DENSE'

if dct == "DOG":
    detector = FeatureDetector_create('SIFT')
elif dct == "FASTHESSIAN":
    detector = FeatureDetector_create('SURF')
else:
    detector = FeatureDetector_create(dct)

# extractor
# extr = 'SIFT'
extr = 'RootSIFT'
# extr = 'SURF'
# extr = 'BRIEF'
# extr = 'ORB'
# extr = 'BRISK'
extractor = DescriptorExtractor_create(extr)

imgA = img_proc(train_img)
kpsA = detector.detect(imgA)
(kpsA, featuresA) = extractor.compute(imgA, kpsA)

len(kpsA)

flag = 2 #cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# flag = 4 #cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
img_keypoints = cv2.drawKeypoints(train_img, kpsA, None, (255, 0, 0), flag)
cv2.imshow("Keypoints", img_keypoints)

# cv2.waitKey(0)
cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------------------------
# keypoint matching
# ------------------------------------------------------------------------------------------------------

# matc = 'BruteForce-Hamming'
# matc = 'BruteForce'
# matc = 'BruteForce-SL2'
matc = 'BruteForce-L1'
# matc = 'FlannBased'
matcher = DescriptorMatcher_create(matc)


# ----------
# query image: mp4:  non-champ
# vid2 = cv2.VideoCapture(video_file1)
# same frame:
# vid2 = cv2.VideoCapture(video_file3)
# start_frame2 = 10
# end_frame2 = start_frame2 + 1

vid2 = cv2.VideoCapture(video_file)

for i in range(len(screw_time_list)):
    start_frame2 = screw_time_list[i][0]
    end_frame2 = screw_time_list[i][1]

    vid2.set(1, start_frame2)


    j2 = start_frame2

    while j2 <= end_frame2:

        is_read, frame = vid2.read()
        if not (is_read):
            break
        frame4 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
        frame4 = frame4[10:120, 90:320]
        imgB = img_proc(frame4)
        kpsB = detector.detect(imgB)
        (kpsB, featuresB) = extractor.compute(imgB, kpsB)

        matches = []
        k = 2
        # k = 4
        rawMatches = matcher.knnMatch(featuresA, featuresB, k)
        if rawMatches is not None:
            for m in rawMatches:
                if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                    matches.append((m[0].trainIdx, m[0].queryIdx))

            (hA, wA) = train_img.shape[:2]
            (hB, wB) = frame4.shape[:2]
            vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
            vis[0:hA, 0:wA] = train_img
            vis[0:hA, wA:] = frame4

            for (trainIdx, queryIdx) in matches:
                color = np.random.randint(0, high=255, size=(3,))
                color = tuple(map(int, color))
                ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
                ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
                radius = 4
                BLUE = (255, 0, 0)
                thickness = 1
                cv2.circle(vis, ptA, radius, BLUE, thickness)
                cv2.circle(vis, ptB, radius, BLUE, thickness)
                # cv2.line(vis, ptA, ptB, BLUE, 1)
                cv2.line(vis, ptA, ptB, color, 1)

        cv2.imshow('key points matching', vis)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

        j2 += 1
