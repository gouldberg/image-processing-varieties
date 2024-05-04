
import numpy as np
import cv2
import imutils



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
    kernel = np.ones((3, 3), np.uint8)
    # img_obj = adjust_gamma(img, gamma=1.75)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (3, 3), 0)
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


# ----------
# Single Gaussian:
# For every pixel, fit one Gaussian PDF distribution (µ,σ) on the most recent n frames
# (this gives the background PDF).
# To accommodate for change in background over time (e.g. due to illumination changes or non-static background objects),
# at every frame, every pixel's mean and variance must be updated.

def single_gaussian(frame_gray, mean, var, alpha=0.25):
    new_mean = (1 - alpha) * mean + alpha * frame_gray
    new_mean = new_mean.astype(np.uint8)
    new_var = (alpha) * (cv2.subtract(frame_gray, mean) ** 2) + (1 - alpha) * (var)

    value = cv2.absdiff(frame_gray, mean)
    value = value / np.sqrt(var)

    mean = np.where(value < 2.5, new_mean, mean)
    var = np.where(value < 2.5, new_var, var)
    a = np.uint8([255])
    b = np.uint8([0])
    background = np.where(value < 2.5, frame_gray, 0)
    foreground = np.where(value >= 2.5, frame_gray, b)

    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(foreground, kernel, iterations=2)
    # erode = cv2.absdiff(foreground, background)

    return erode, new_mean, new_var


# ------------------------------------------------------------------------------------------------------
# background subtraction
# ------------------------------------------------------------------------------------------------------

video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly'

# ----------
# non-champion
vid_file1 = 'Gamma_Assembly_192_168_32_69_1_20201130110000_10min.mp4'
video_file1 = os.path.join(video_orig_path, vid_file1)

vid_file2 = 'Gamma_Assembly_192_168_32_69_1_20201130180000_10min.mp4'
video_file2 = os.path.join(video_orig_path, vid_file2)

# champion
vid_file3 = 'Gamma_Assembly_192_168_32_69_1_20210223133400_10min.mp4'
video_file3 = os.path.join(video_orig_path, vid_file3)

vid_file4 = 'Gamma_Assembly_192_168_32_69_1_20210303161000_10min.mp4'
video_file4 = os.path.join(video_orig_path, vid_file4)


# ----------
# 0419 - 0421
vid_file5 = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file5 = os.path.join(video_orig_path, vid_file5)

vid_file6 = 'test_192_168_32_70_20210421073455_60min.mp4'
video_file6 = os.path.join(video_orig_path, vid_file6)


# video_file_tmp = 'C:\\Users\\kosei-wada\\Desktop\\reference\\ManyVisitors_1.avi'
video_file_tmp = 'C:\\Users\\kosei-wada\\Desktop\\reference\\ThreeVisitors.avi'


# rate = 4
rate = 1
fps = 30


def img_proc(img):
    kernel = np.ones((3, 3), np.uint8)
    # img_obj = adjust_gamma(img, gamma=1.75)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (5, 5), 0)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj

# ----------
# BackgroundSubtractorGMG
# It uses first few (120 by default) frames for background modelling.
# It employs probabilistic foreground segmentation algorithm that identifies possible foreground objects
# using Bayesian inference.
# The estimates are adaptive; newer observations are more heavily weighted than old observations
# to accommodate variable illumination.
# Several morphological filtering operations like closing and opening are done to remove unwanted noise.
# You will get a black window during first few frames.

# decisionThreshold = 0.95
# fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=5,
#                                                     decisionThreshold=decisionThreshold)

decisionThreshold = 0.6
fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=5,
                                                    decisionThreshold=decisionThreshold)

# ----------
# MOG:  Gaussian Mixture-based Background/Foreground Segmentation
# It uses a method to model each background pixel by a mixture of K Gaussian distributions (K = 3 to 5).
# The weights of the mixture represent the time proportions that those colours stay in the scene.
# The probable background colours are the ones which stay longer and more static.

# noiseSigma should be large
noiseSigma=20
fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG(history=5, nmixtures=25,
                                                    backgroundRatio=0.2, noiseSigma=noiseSigma)

# ----------
# MOG2
fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=5)
fgbg_mog2.setBackgroundRatio(0.2)


# ----------
# CNT: useHistory=False is better
fgbg_cnt = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=5, maxPixelStability=5,
                                                    useHistory=False)


# ----------
# vid = cv2.VideoCapture(video_file5)
# vid = cv2.VideoCapture(video_file6)
vid = cv2.VideoCapture(video_file_tmp)

# head is visible
# start = 70
# end = 100

# screw driver is moving, human shadow is visible
start = 190
end = 240

# start = 880
# end = 915

# start = 1320
# end = 1345

# human shadow is visible
# start = 2790
# end = 2820

# start = 3215
# end = 3245

# start = 3285
# end = 3400

start = 0
end = 95
start_frame = start * fps
end_frame = end * fps

vid.set(1, start_frame)
j = start_frame

lst_gmg, lst_mog, lst_mog2, lst_cnt, lst_sg = [], [], [], [], []

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

while j <= end_frame:
    print(j)
    is_read, frame = vid.read()
    if not (is_read):
        break

    frame2 = cv2.resize(frame, (int(frame.shape[1] * rate), int(frame.shape[0] * rate)))
    # frame2 = img_proc(frame2)

    # if j == start_frame:
    #     var = np.ones((frame2.shape[0], frame2.shape[1]), np.uint8)
    #     var[:frame2.shape[0], :frame2.shape[1]] = 150
    #     mean = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    out_gmg = fgbg_gmg.apply(frame2)
    # out_mog = fgbg_mog.apply(frame2)
    # out_mog2 = fgbg_mog2.apply(frame2)
    # out_cnt = fgbg_cnt.apply(frame2)

    # out_sg, mean, var = single_gaussian(frame_gray=cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
    #                          mean=mean, var=var, alpha=0.25)

    out_gmg = cv2.morphologyEx(out_gmg, cv2.MORPH_OPEN, kernel)

    # out_mog = cv2.morphologyEx(out_mog, cv2.MORPH_OPEN, kernel)
    # out_mog2 = cv2.morphologyEx(out_mog2, cv2.MORPH_OPEN, kernel)
    # out_cnt2 = cv2.morphologyEx(out_cnt, cv2.MORPH_OPEN, kernel)

    # for MOG2, threshold >= 200 is better to remove shadow
    # _, out_mog2_2 = cv2.threshold(out_mog2, 250, 255, cv2.THRESH_BINARY)

    # lst_gmg.extend([sum(sum(out_gmg))/(480*rate*640*rate)])
    # lst_mog.extend([sum(sum(out_mog))/(480*rate*640*rate)])
    # lst_mog2.extend([sum(sum(out_mog2_2))/(480*rate*640*rate)])
    # lst_cnt.extend([sum(sum(out_cnt))/(480*rate*640*rate)])
    # lst_sg.extend([sum(sum(out_sg))/(480*rate*640*rate)])

    cv2.imshow('image orig', frame2)
    cv2.imshow('GMG', out_gmg)
    # cv2.imshow('MOG', out_mog)
    # cv2.imshow('MOG2', out_mog2)
    # cv2.imshow('CNT', out_cnt)
    # cv2.imshow('SG', out_sg)

    if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    j += 1

cv2.destroyAllWindows()

vid.release()

lst_val_gmg = pd.DataFrame({'val': lst_gmg})
lst_val_mog = pd.DataFrame({'val': lst_mog})
lst_val_mog2 = pd.DataFrame({'val': lst_mog2})
lst_val_cnt = pd.DataFrame({'val': lst_cnt})
# lst_val_sg = pd.DataFrame({'val': lst_sg})

# ----------
lst_val_gmg.val.plot()
lst_val_cnt.val.plot()

lst_val_mog.val.plot()
lst_val_mog2.val.plot()

# lst_val_sg.val.plot()


# ------------------------------------------------------------------------------------------------------
# image differencing
# ------------------------------------------------------------------------------------------------------

video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly'

# ----------
# non-champion
vid_file1 = 'Gamma_Assembly_192_168_32_69_1_20201130110000_10min.mp4'
video_file1 = os.path.join(video_orig_path, vid_file1)

vid_file2 = 'Gamma_Assembly_192_168_32_69_1_20201130180000_10min.mp4'
video_file2 = os.path.join(video_orig_path, vid_file2)

# champion
vid_file3 = 'Gamma_Assembly_192_168_32_69_1_20210223133400_10min.mp4'
video_file3 = os.path.join(video_orig_path, vid_file3)

vid_file4 = 'Gamma_Assembly_192_168_32_69_1_20210303161000_10min.mp4'
video_file4 = os.path.join(video_orig_path, vid_file4)


# ----------
# 0419 - 0421
vid_file5 = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file5 = os.path.join(video_orig_path, vid_file5)


# rate = 4
rate = 1


# ----------
# imgA
vid = cv2.VideoCapture(video_file3)
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



# ----------
# calc image differences

frame2Gray = img_proc(frame2)
frame3Gray = img_proc(frame3)

frame_dif = np.abs(frame3Gray - frame2Gray)
frame_dif = np.where(frame_dif > 10, 0, 255)
frame_dif = np.array(frame_dif, dtype='uint8')

cv2.imshow('image dif', frame_dif)
cv2.imshow('image3', frame3Gray)
cv2.imshow('image2', frame2Gray)
