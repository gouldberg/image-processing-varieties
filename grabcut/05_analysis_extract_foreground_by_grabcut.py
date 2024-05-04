
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


def img_proc(img):
    # kernel = np.ones((3, 3), np.uint8)
    img_obj = adjust_gamma(img, gamma=1.75)
    # img_obj = adjust_gamma(img, gamma=1.0)
    # img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    # img_obj = cv2.GaussianBlur(img_obj, (3, 3), 0)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # # img_obj = cv2.Canny(img_obj, 50, 50)
    # img_obj = sub_color(img_obj, K=20)
    return img_obj


# ------------------------------------------------------------------------------------------------------
# GrabCut
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
frame2 = img_proc(frame2)
frame3 = img_proc(frame3)

cv2.imwrite(os.path.join(base_path, 'tmp.png'), frame2)


# ----------
# ----------
mask = cv2.imread(os.path.join(base_path, 'tmp.png'), cv2.GC_BGD)

# any mask values greater than zero should be set to probable foreground
# mask[mask > 0] = cv2.GC_PR_FGD
# mask[mask <= 0] = cv2.GC_BGD

mask[((mask >= 150)&(mask <= 180))] = cv2.GC_PR_FGD
mask[((mask < 150)|(mask > 180))] = cv2.GC_BGD


# allocate memory for two arrays that the GrabCut algorithm internally uses when segmenting the foreground from the background
fgModel = np.zeros((1,65), dtype="float")
bgModel = np.zeros((1,65), dtype="float")


# ----------
# grab cut
# (mask, bgModel, fgModel) = cv2.grabCut(frame3, mask, None, bgModel, fgModel,
#                                        iterCount=20, mode=cv2.GC_INIT_WITH_MASK)

(mask, bgModel, fgModel) = cv2.grabCut(frame3, mask, None, bgModel, fgModel,
                                       iterCount=20, mode=2)

print(np.unique(mask, return_counts=True))


# ----------
# the output mask has for possible output values, marking each pixel in the mask as
# (1) definite background, (2) definite foreground,
# (3) probable background, and (4) probable foreground

values = (
    ("Definite Background", cv2.GC_BGD),
    ("Probable Background", cv2.GC_PR_BGD),
    ("Definite Foreground", cv2.GC_FGD),
    ("Probable Foreground", cv2.GC_PR_FGD),
)

# loop over the possible GrabCut mask values
for (name ,value) in values:
    print("[INFO] showing mask for ' {}'".format(name))
    valueMask = (mask == value).astype("uint8") * 255

    cv2.imshow(name, valueMask)
    cv2.waitKey(0)


# ----------
# display mask
outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
outputMask = (outputMask * 255).astype("uint8")

output = cv2.bitwise_and(frame3, frame3, mask=outputMask)

cv2.imshow("Input3", frame3)
cv2.imshow("Input2", frame2)
cv2.imshow("GrabCut Mask", outputMask)
cv2.imshow("GrabCut Output", output)
cv2.waitKey(0)



