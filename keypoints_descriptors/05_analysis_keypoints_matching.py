
import numpy as np
import cv2
import imutils


# ------------------------------------------------------------------------------------------------------
# support functions
# ------------------------------------------------------------------------------------------------------

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
# keypoints matching
# ------------------------------------------------------------------------------------------------------

from typing import Tuple, Optional, List, Sequence

Point = Tuple[float, float]

# class RootSIFT:
#     def __init__(self):
#         # initialize the SIFT feature extractor for OpenCV 2.4
#         if imutils.is_cv2():
#             self.extractor = cv2.DescriptorExtractor_create("SIFT")
#
#         # otherwise, initialize the SIFT feature extractor for OpenCV 3+
#         else:
#             self.extractor = cv2.xfeatures2d.SIFT_create()
#
#     def compute(self, image, kps, eps=1e-7):
#         # compute SIFT descriptors for OpenCV 2.4
#         if imutils.is_cv2:
#             (kps, descs) = self.extractor.compute(image, kps)
#
#         # otherwise, computer SIFT descriptors for OpenCV 3+
#         else:
#             (kps, descs) = self.extractor.detectAndCompute(image, None)
#
#         # if there are no keypoints or descriptors, return an empty tuple
#         if len(kps) == 0:
#             return ([], None)
#
#         # apply the Hellinger kernel by first L1-normalizing and taking the square-root
#         descs /= (descs.sum(axis=1, keepdims=True) + eps)
#         descs = np.sqrt(descs)
#
#         # return a tuple of the keypoints and descriptors
#         return (kps, descs)


class Outlier(Exception):
    pass

def img_proc(img):
    kernel = np.ones((5, 5), np.uint8)
    # img_obj = adjust_gamma(img, gamma=1)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (5, 5), 0)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    img_obj = auto_canny(img_obj)
    return img_obj

class FeatureMatching:

    def __init__(self, train_image: np.ndarray):

        # ----------
        # extractor
        # self.f_extractor = cv2.xfeatures2d.SIFT_create()
        # self.f_extractor = cv2.ORB_create()

        self.detector = cv2.FastFeatureDetector_create()
        self.f_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # ----------
        # template image: "train" image
        self.img_obj = img_proc(train_image)
        self.sh_train = self.img_obj.shape[:2]

        # ----------
        # feature extraction
        # self.key_train, self.desc_train = \
        #     self.f_extractor.detectAndCompute(self.img_obj, None)
        self.key_train, self.desc_train = \
            self.f_extractor.compute(self.img_obj, self.detector.detect(self.img_obj))
        if len(self.key_train) == 0:
            self.key_train = []
            self.desc_train = None

        # ----------
        # apply the Hellinger kernel by first L1-normalizing and taking the square-root
        # self.eps = 1e-3
        # self.eps = 1e-7
        # self.desc_train /= (self.desc_train.sum(axis=1, keepdims=True) + self.eps)
        # self.desc_train = np.sqrt(self.desc_train)

        # ----------
        # matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        index_params = {"algorithm": 0, "trees": 5}
        search_params = {"checks": 50}
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # self.flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        # initialize tracking
        self.last_hinv = np.zeros((3, 3))
        self.max_error_hinv = 50.
        self.num_frames_no_success = 0
        self.max_frames_no_success = 5

    def match(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detects and tracks an object of interest in a video frame
            This method detects and tracks an object of interest (of which a
            SURF descriptor was obtained upon initialization) in a video frame.
            Correspondence is established with a FLANN based matcher.
            The algorithm then applies a perspective transform on the frame in
            order to project the object of interest to the frontal plane.
            Outlier rejection is applied to improve the tracking of the object
            from frame to frame.
            :param frame: input (query) image in which to detect the object
            :returns: (success, frame) whether the detection was successful and
                      and the perspective-transformed frame
        """

        # ----------
        img_query = img_proc(frame)
        sh_query = img_query.shape  # rows,cols

        # ----------
        # feature extraction
        # key_query, desc_query = \
        #     self.f_extractor.detectAndCompute(img_query, None)
        key_query, desc_query = \
            self.f_extractor.compute(img_query, self.detector.detect(self.img_obj))
        if len(key_query) == 0:
            key_query = []
            desc_query = None
        # desc_query /= (desc_query.sum(axis=1, keepdims=True) + self.eps)
        # desc_query = np.sqrt(desc_query)

        # ----------
        # find best matches (kNN) and discard bad matches (ratio test as per Lowe's paper, thresh=0.8)
        # k = 2
        k = 4
        matches = self.flann.knnMatch(self.desc_train, desc_query, k=k)

        # thresh = 0.7
        thresh = 0.95
        good_matches = [x[0] for x in matches if x[0].distance < thresh * x[1].distance]
        train_points = [self.key_train[good_match.queryIdx].pt for good_match in good_matches]
        query_points = [key_query[good_match.trainIdx].pt for good_match in good_matches]

        # try:
        #     # early outlier detection and rejection
        #     if len(good_matches) < 4:
        #         raise Outlier("Too few matches")
        #
        #     # ----------
        #     # corner point detection
        #     # calculates the homography matrix needed to convert between keypoints from the train image and the query image
        #     dst_corners = detect_corner_points(train_points, query_points, self.sh_train)
        #
        #     # if any corners lie significantly outside the image, skip frame
        #     if np.any((dst_corners < -20) | (dst_corners > np.array(sh_query) + 20)):
        #         raise Outlier("Out of image")
        #
        #     # find the area of the quadrilateral that the four corner points spans
        #     area = 0
        #     for prev, nxt in zip(dst_corners, np.roll(dst_corners, -1, axis=0)):
        #         area += (prev[0] * nxt[1] - prev[1] * nxt[0]) / 2.
        #
        #     # reject corner points if area is unreasonable
        #     if not np.prod(sh_query) / 16. < area < np.prod(sh_query) / 2.:
        #         raise Outlier("Area is unreasonably small or large")
        #
        #     # bring object of interest to frontal plane
        #     train_points_scaled = self.scale_and_offset(train_points, self.sh_train, sh_query)
        #     Hinv, _ = cv2.findHomography(np.array(query_points), np.array(train_points_scaled), cv2.RANSAC)
        #
        #     # if last frame recent: new Hinv must be similar to last one
        #     # else: accept whatever Hinv is found at this point
        #     similar = np.linalg.norm(Hinv - self.last_hinv) < self.max_error_hinv
        #     recent = self.num_frames_no_success < self.max_frames_no_success
        #     if recent and not similar:
        #         raise Outlier("Not similar transformation")
        #
        # except Outlier as e:
        #     self.num_frames_no_success += 1
        #     # return False, None, None
        #     return False, None
        # else:
        #     self.num_frames_no_success = 0
        #     self.last_h = Hinv

        # outline corner points of train image in query image
        # img_warped = cv2.warpPerspective(img_query, Hinv, (sh_query[1], sh_query[0]))
        img_flann = draw_good_matches(self.img_obj, self.key_train, img_query, key_query, good_matches)
        # adjust x-coordinate (col) of corner points so that they can be drawn next to the train image (add self.sh_train[1])
        # dst_corners[:, 0] += self.sh_train[1]
        # cv2.polylines(img_flann, [dst_corners.astype(np.int)], isClosed=True, color=(0, 255, 0), thickness=3)

        # return True, img_warped, img_flann
        return True, img_flann

    @staticmethod
    def scale_and_offset(points: Sequence[Point], source_size: Tuple[int, int], dst_size: Tuple[int, int], factor: float = 0.5) -> List[Point]:
        dst_size = np.array(dst_size)
        scale = 1 / np.array(source_size) * dst_size * factor
        bias = dst_size * (1 - factor) / 2
        return [tuple(np.array(pt) * scale + bias) for pt in points]


def detect_corner_points(src_points: Sequence[Point], dst_points: Sequence[Point], sh_src: Tuple[int, int]) -> np.ndarray:
    """Detects corner points in an input (query) image
        This method finds the homography matrix to go from the template
        (train) image to the input (query) image, and finds the coordinates
        of the good matches (from the train image) in the query image.
        :param key_frame: keypoints of the query image
        :param good_matches: list of good matches
        :returns: coordinates of good matches in transformed query image
    """

    # -----------
    # find homography using RANSAC
    H, _ = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC)
    if H is None:
        raise Outlier("Homography not found")

    # -----------
    height, width = sh_src
    src_corners = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)

    return cv2.perspectiveTransform(src_corners[None, :, :], H)[0]


def draw_good_matches(img1: np.ndarray,
                      kp1: Sequence[cv2.KeyPoint],
                      img2: np.ndarray,
                      kp2: Sequence[cv2.KeyPoint],
                      matches: Sequence[cv2.DMatch]) -> np.ndarray:
    """Visualizes a list of good matches
        This function visualizes a list of good matches. It is only required in
        OpenCV releases that do not ship with the function drawKeypoints.
        The function draws two images (img1 and img2) side-by-side,
        highlighting a list of keypoints in both, and connects matching
        keypoints in the two images with blue lines.
        :param img1: first image
        :param kp1: list of keypoints for first image
        :param img2: second image
        :param kp2: list of keypoints for second image
        :param matches: list of good matches
        :returns: annotated output image
    """
    # Create a new output image of a size that will fit the two images together
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = img1[..., None]
    out[:rows2, cols1:cols1 + cols2, :] = img2[..., None]

    # For each pair of points we have between both images, draw circles, then connect a line between them

    for m in matches:
        # Get the matching keypoints for each of the images and convert them to int
        c1 = tuple(map(int, kp1[m.queryIdx].pt))
        c2 = tuple(map(int, kp2[m.trainIdx].pt))
        c2 = c2[0] + cols1, c2[1]
        radius = 4
        BLUE = (255, 0, 0)
        thickness = 1
        cv2.circle(out, c1, radius, BLUE, thickness)
        cv2.circle(out, c2, radius, BLUE, thickness)
        cv2.line(out, c1, c2, BLUE, thickness)

    return out


# ----------
base_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis'

# vid_file = 'Gamma_Assembly_192_168_32_69_1_20201130065000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20201130110000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20201130180000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20201132000000_10min_result.mp4'
#
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210223133400_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210223134400_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210223135400_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210223140400_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210223141400_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210223142400_10min_result.mp4'
#
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303160000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303161000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303162000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303163000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303164000_10min_result.mp4'
# # 165000_10min includes no light pipe operation
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303165000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303170000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303171000_10min_result.mp4'
# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303172000_10min_result.mp4'
#
#
# # ----------
# # non-champion
# vid_file1 = 'Gamma_Assembly_192_168_32_69_1_20201130110000_10min_result.mp4'
# video_file1 = os.path.join('C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly', vid_file1)
#
# vid_file2 = 'Gamma_Assembly_192_168_32_69_1_20201130180000_10min_result.mp4'
# video_file2 = os.path.join('C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly', vid_file2)
#
# # champion
# vid_file3 = 'Gamma_Assembly_192_168_32_69_1_20210223133400_10min_result.mp4'
# video_file3 = os.path.join('C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly', vid_file3)
#
# vid_file4 = 'Gamma_Assembly_192_168_32_69_1_20210303161000_10min_result.mp4'
# video_file4 = os.path.join('C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly', vid_file4)


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

for i in range(len(screw_time_list)):
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
        # frame3 = frame2[15:120, 90:320]
        frame3 = frame2
        cv2.imshow('image', frame3)
        print(j)
        if cv2.waitKey(3) & 0xFF == ord('q'):
                break
        j += 1

cv2.destroyAllWindows()


# ----------
# keypoints from reference image
train_img = frame3
matching = FeatureMatching(train_img)

print(len(matching.key_train))
print(matching.desc_train.shape)

flag = 2 #cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# flag = 4 #cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
img_keypoints = cv2.drawKeypoints(train_img, matching.key_train, None, (255, 0, 0), flag)
cv2.imshow("Keypoints", img_keypoints)

# cv2.waitKey(0)
cv2.destroyAllWindows()


# ----------
# query image: mp4
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
        frame3 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
        # frame3 = frame2[15:120, 90:320]

        cv2.imshow("frame", frame3)
        # match_succsess, img_warped, img_flann = matching.match(frame3)
        match_succsess, img_flann = matching.match(frame3)
        if match_succsess:
            # cv2.imshow("res", img_warped)
            cv2.imshow("flann", img_flann)
        if cv2.waitKey(1) & 0xff == 27:
            break

        j2 += 1


#########################################
# extractor
# f_extractor = cv2.xfeatures2d.SIFT_create()
f_extractor = cv2.ORB_create()

# detector = cv2.FastFeatureDetector_create()
# f_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# ----------
# template image: "train" image
img_obj = img_proc(train_img)
sh_train = img_obj.shape[:2]

# ----------
# feature extraction
key_train, desc_train = \
    f_extractor.detectAndCompute(img_obj, None)
# key_train, desc_train = \
#     f_extractor.compute(img_obj, detector.detect(img_obj))

if len(key_train) == 0:
    key_train = []
    desc_train = None

# ----------
# apply the Hellinger kernel by first L1-normalizing and taking the square-root
# eps = 1e-3
# eps = 1e-7
# desc_train /= (desc_train.sum(axis=1, keepdims=True) + eps)
# desc_train = np.sqrt(desc_train)

# ----------
# matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
index_params = {"algorithm": 0, "trees": 5}
search_params = {"checks": 50}
flann = cv2.FlannBasedMatcher(index_params, search_params)
# flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

# initialize tracking
last_hinv = np.zeros((3, 3))
max_error_hinv = 50.
num_frames_no_success = 0
max_frames_no_success = 5


#########################################
img_query = img_proc(frame3)
sh_query = img_query.shape  # rows,cols

# ----------
# feature extraction
key_query, desc_query = \
    f_extractor.detectAndCompute(img_query, None)
# key_query, desc_query = \
#     f_extractor.compute(img_query, detector.detect(img_obj))
if len(key_query) == 0:
    key_query = []
    desc_query = None
# desc_query /= (desc_query.sum(axis=1, keepdims=True) + eps)
# desc_query = np.sqrt(desc_query)

# ----------
# find best matches (kNN) and discard bad matches (ratio test as per Lowe's paper, thresh=0.8)
# k = 2
k = 4
matches = flann.knnMatch(desc_train, desc_query, k=k)

# thresh = 0.7
thresh = 0.95
good_matches = [x[0] for x in matches if x[0].distance < thresh * x[1].distance]
train_points = [key_train[good_match.queryIdx].pt for good_match in good_matches]
query_points = [key_query[good_match.trainIdx].pt for good_match in good_matches]

# try:
#     # early outlier detection and rejection
#     if len(good_matches) < 4:
#         raise Outlier("Too few matches")
#
#     # ----------
#     # corner point detection
#     # calculates the homography matrix needed to convert between keypoints from the train image and the query image
#     dst_corners = detect_corner_points(train_points, query_points, self.sh_train)
#
#     # if any corners lie significantly outside the image, skip frame
#     if np.any((dst_corners < -20) | (dst_corners > np.array(sh_query) + 20)):
#         raise Outlier("Out of image")
#
#     # find the area of the quadrilateral that the four corner points spans
#     area = 0
#     for prev, nxt in zip(dst_corners, np.roll(dst_corners, -1, axis=0)):
#         area += (prev[0] * nxt[1] - prev[1] * nxt[0]) / 2.
#
#     # reject corner points if area is unreasonable
#     if not np.prod(sh_query) / 16. < area < np.prod(sh_query) / 2.:
#         raise Outlier("Area is unreasonably small or large")
#
#     # bring object of interest to frontal plane
#     train_points_scaled = self.scale_and_offset(train_points, self.sh_train, sh_query)
#     Hinv, _ = cv2.findHomography(np.array(query_points), np.array(train_points_scaled), cv2.RANSAC)
#
#     # if last frame recent: new Hinv must be similar to last one
#     # else: accept whatever Hinv is found at this point
#     similar = np.linalg.norm(Hinv - self.last_hinv) < self.max_error_hinv
#     recent = self.num_frames_no_success < self.max_frames_no_success
#     if recent and not similar:
#         raise Outlier("Not similar transformation")
#
# except Outlier as e:
#     self.num_frames_no_success += 1
#     # return False, None, None
#     return False, None
# else:
#     self.num_frames_no_success = 0
#     self.last_h = Hinv

# outline corner points of train image in query image
# img_warped = cv2.warpPerspective(img_query, Hinv, (sh_query[1], sh_query[0]))
img_flann = draw_good_matches(img_obj, key_train, img_query, key_query, good_matches)
# adjust x-coordinate (col) of corner points so that they can be drawn next to the train image (add self.sh_train[1])
# dst_corners[:, 0] += sh_train[1]
# cv2.polylines(img_flann, [dst_corners.astype(np.int)], isClosed=True, color=(0, 255, 0), thickness=3)
