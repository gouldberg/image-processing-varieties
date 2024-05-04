
import numpy as np
import cv2
import imutils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import wx

base_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis'


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
    # kernel = np.ones((3, 3), np.uint8)
    img_obj = adjust_gamma(img, gamma=1.0)
    # img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    # img_obj = cv2.GaussianBlur(img_obj, (3, 3), 0)
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
# class SceneReconstruction3D
# ------------------------------------------------------------------------------------------------------

class SceneReconstruction3D:
    """3D scene reconstruction

        This class implements an algorithm for 3D scene reconstruction using
        stereo vision and structure-from-motion techniques.

        A 3D scene is reconstructed from a pair of images that show the same
        real-world scene from two different viewpoints. Feature matching is
        performed either with rich feature descriptors or based on optic flow.
        3D coordinates are obtained via triangulation.

        Note that a complete structure-from-motion pipeline typically includes
        bundle adjustment and geometry fitting, which are out of scope for
        this project.
    """

    def __init__(self, K, dist):
        """Constructor

            This method initializes the scene reconstruction algorithm.

            :param K: 3x3 intrinsic camera matrix
            :param dist: vector of distortion coefficients
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)  # store inverse for fast access
        self.d = dist

    def load_image_pair(
            self,
            img_path1: str,
            img_path2: str,
            use_pyr_down: bool = True) -> None:

        self.img1, self.img2 = [
            cv2.undistort(
                self.load_image(
                    path, use_pyr_down), self.K, self.d) for path in (
                img_path1, img_path2)]

    @staticmethod
    def load_image(
            img_path: str,
            use_pyr_down: bool,
            target_width: int = 600) -> np.ndarray:
        """Loads pair of images

            This method loads the two images for which the 3D scene should be
            reconstructed. The two images should show the same real-world scene
            from two different viewpoints.

            :param img_path1: path to first image
            :param img_path2: path to second image
            :param use_pyr_down: flag whether to downscale the images to
                                 roughly 600px width (True) or not (False)
        """

        img = cv2.imread(img_path, cv2.CV_8UC3)

        # make sure image is valid
        assert img is not None, f"Image {img_path} could not be loaded."
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # scale down image if necessary
        while use_pyr_down and img.shape[1] > 2 * target_width:
            img = cv2.pyrDown(img)
        return img

    def plot_optic_flow(self):
        """Plots optic flow field

            This method plots the optic flow between the first and second
            image.
        """
        self._extract_keypoints_flow()

        img = np.copy(self.img1)
        for pt1, pt2 in zip(self.match_pts1, self.match_pts2):
            cv2.arrowedLine(img, tuple(pt1), tuple(pt2),
                            color=(255, 0, 0))

        cv2.imshow("imgFlow", img)
        cv2.waitKey()

    def draw_epipolar_lines(self, feat_mode: str = "SIFT"):
        """Draws epipolar lines

            This method computes and draws the epipolar lines of the two
            loaded images.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("sift") or optic flow ("flow")
        """
        self._extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        # Find epilines corresponding to points in right image (second image)
        # and drawing its lines on left image
        pts2re = self.match_pts2.reshape(-1, 1, 2)
        lines1 = cv2.computeCorrespondEpilines(pts2re, 2, self.F)
        lines1 = lines1.reshape(-1, 3)
        img3, img4 = self._draw_epipolar_lines_helper(self.img1, self.img2,
                                                      lines1, self.match_pts1,
                                                      self.match_pts2)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        pts1re = self.match_pts1.reshape(-1, 1, 2)
        lines2 = cv2.computeCorrespondEpilines(pts1re, 1, self.F)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = self._draw_epipolar_lines_helper(self.img2, self.img1,
                                                      lines2, self.match_pts2,
                                                      self.match_pts1)

        cv2.imshow("left", img1)
        cv2.imshow("right", img3)
        cv2.waitKey()

    def plot_rectified_images(self, feat_mode: str = "SIFT"):
        """Plots rectified images

            This method computes and plots a rectified version of the two
            images side by side.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("sift") or optic flow ("flow")
        """
        self._extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        R = self.Rt2[:, :3]
        T = self.Rt2[:, 3]
        # perform the rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.K, self.d,
                                                          self.K, self.d,
                                                          self.img1.shape[:2],
                                                          R, T, alpha=1.0)
        mapx1, mapy1 = cv2.initUndistortRectifyMap(self.K, self.d, R1, self.K,
                                                   self.img1.shape[:2],
                                                   cv2.CV_32F)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(self.K, self.d, R2, self.K,
                                                   self.img2.shape[:2],
                                                   cv2.CV_32F)
        img_rect1 = cv2.remap(self.img1, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(self.img2, mapx2, mapy2, cv2.INTER_LINEAR)

        # draw the images side by side
        total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                      img_rect1.shape[1] + img_rect2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
        img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

        # draw horizontal lines every 25 px accross the side by side image
        for i in range(20, img.shape[0], 25):
            cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

        cv2.imshow('imgRectified', img)
        cv2.waitKey()

    def plot_point_cloud(self, feat_mode="sift"):
        """Plots 3D point cloud

            This method generates and plots a 3D point cloud of the recovered
            3D scene.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("sift") or optic flow ("flow")
        """
        self._extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        # triangulate points
        first_inliers = np.array(self.match_inliers1)[:, :2]
        second_inliers = np.array(self.match_inliers2)[:, :2]
        pts4D = cv2.triangulatePoints(self.Rt1, self.Rt2, first_inliers.T,
                                      second_inliers.T).T

        # convert from homogeneous coordinates to 3D
        pts3D = pts4D[:, :3] / pts4D[:, 3, None]

        # plot with matplotlib
        Xs, Zs, Ys = [pts3D[:, i] for i in range(3)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xs, Ys, Zs, c=Ys,cmap=cm.hsv, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D point cloud: Use pan axes button below to inspect')
        plt.show()

    def _extract_keypoints(self, feat_mode):
        """Extracts keypoints

            This method extracts keypoints for feature matching based on
            a specified mode:
            - "sift": use rich sift descriptor
            - "flow": use optic flow

            :param feat_mode: keypoint extraction mode ("sift" or "flow")
        """
        # extract features
        if feat_mode.lower() == "sift":
            # feature matching via sift and BFMatcher
            self._extract_keypoints_sift()
        elif feat_mode.lower() == "flow":
            # feature matching via optic flow
            self._extract_keypoints_flow()
        else:
            sys.exit(f"Unknown feat_mode {feat_mode}. Use 'sift' or 'FLOW'")

    def _extract_keypoints_sift(self):
        """Extracts keypoints via sift descriptors"""
        # extract keypoints and descriptors from both images
        # detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.11, edgeThreshold=10)
        detector = cv2.xfeatures2d.SIFT_create()
        first_key_points, first_desc = detector.detectAndCompute(self.img1,
                                                                 None)
        second_key_points, second_desc = detector.detectAndCompute(self.img2,
                                                                   None)
        # match descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L1, True)
        matches = matcher.match(first_desc, second_desc)

        # generate lists of point correspondences
        self.match_pts1 = np.array(
            [first_key_points[match.queryIdx].pt for match in matches])
        self.match_pts2 = np.array(
            [second_key_points[match.trainIdx].pt for match in matches])

    def _extract_keypoints_flow(self):
        """Extracts keypoints via optic flow"""
        # find FAST features
        fast = cv2.FastFeatureDetector_create()
        first_key_points = fast.detect(self.img1)

        first_key_list = [i.pt for i in first_key_points]
        first_key_arr = np.array(first_key_list).astype(np.float32)

        second_key_arr, status, err = cv2.calcOpticalFlowPyrLK(
            self.img1, self.img2, first_key_arr, None)

        # filter out the points with high error
        # keep only entries with status=1 and small error
        condition = (status == 1) * (err < 5.)
        concat = np.concatenate((condition, condition), axis=1)
        first_match_points = first_key_arr[concat].reshape(-1, 2)
        second_match_points = second_key_arr[concat].reshape(-1, 2)

        self.match_pts1 = first_match_points
        self.match_pts2 = second_match_points

    def _find_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC, 0.1, 0.99)

    def _find_essential_matrix(self):
        """Estimates essential matrix based on fundamental matrix """
        self.E = self.K.T.dot(self.F).dot(self.K)

    def _find_camera_matrices_rt(self):
        """Finds the [R|t] camera matrix"""
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0]).reshape(3, 3)

        # iterate over all point correspondences used in the estimation of the
        # fundamental matrix
        first_inliers = []
        second_inliers = []
        for pt1, pt2, mask in zip(
                self.match_pts1, self.match_pts2, self.Fmask):
            if mask:
                # normalize and homogenize the image coordinates
                first_inliers.append(self.K_inv.dot([pt1[0], pt1[1], 1.0]))
                second_inliers.append(self.K_inv.dot([pt2[0], pt2[1], 1.0]))

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in
        # front of both cameras

        R = T = None
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]
        for r in (U.dot(W).dot(Vt), U.dot(W.T).dot(Vt)):
            for t in (U[:, 2], -U[:, 2]):
                if self._in_front_of_both_cameras(
                        first_inliers, second_inliers, r, t):
                    R, T = r, t

        assert R is not None, "Camera matricies were never found"

        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))

    def _draw_epipolar_lines_helper(self, img1, img2, lines, pts1, pts2):
        """Helper method to draw epipolar lines and features """
        if img1.shape[2] == 1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.shape[2] == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        c = img1.shape[1]
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img1, tuple(pt1), 5, color, -1)
            cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    def _in_front_of_both_cameras(self, first_points, second_points, rot,
                                  trans):
        """Determines whether point correspondences are in front of both
           images"""
        print("start")
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :],
                             trans) / np.dot(rot[0, :] - second[0] * rot[2, :],
                                             second)
            first_3d_point = np.array([first[0] * first_z,
                                       second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,
                                                                     trans)

            print(first_3d_point,second_3d_point)
            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True

    def _linear_ls_triangulation(self, u1, P1, u2, P2):
        """Triangulation via Linear-LS method"""
        # build A matrix for homogeneous equation system Ax=0
        # assume X = (x,y,z,1) for Linear-LS method
        # which turns it into AX=B system, where A is 4x3, X is 3x1 & B is 4x1
        A = np.array([u1[0] * P1[2, 0] - P1[0, 0], u1[0] * P1[2, 1] - P1[0, 1],
                      u1[0] * P1[2, 2] - P1[0, 2], u1[1] * P1[2, 0] - P1[1, 0],
                      u1[1] * P1[2, 1] - P1[1, 1], u1[1] * P1[2, 2] - P1[1, 2],
                      u2[0] * P2[2, 0] - P2[0, 0], u2[0] * P2[2, 1] - P2[0, 1],
                      u2[0] * P2[2, 2] - P2[0, 2], u2[1] * P2[2, 0] - P2[1, 0],
                      u2[1] * P2[2, 1] - P2[1, 1],
                      u2[1] * P2[2, 2] - P2[1, 2]]).reshape(4, 3)

        B = np.array([-(u1[0] * P1[2, 3] - P1[0, 3]),
                      -(u1[1] * P1[2, 3] - P1[1, 3]),
                      -(u2[0] * P2[2, 3] - P2[0, 3]),
                      -(u2[1] * P2[2, 3] - P2[1, 3])]).reshape(4, 1)

        ret, X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
        return X.reshape(1, 3)


# ------------------------------------------------------------------------------------------------------
# class BaseLayout
# ------------------------------------------------------------------------------------------------------

"""
A module containing simple GUI layouts using wxPython
This file is heavily based on the work of Michael Beyeler.
"""

class BaseLayout(wx.Frame):
    """ Abstract base class for all layouts in the book.
    A custom layout needs to implement the 2 methods below
        - augment_layout
        - process_frame
    """

    def __init__(self,
                 capture: cv2.VideoCapture,
                 title: str = None,
                 parent=None,
                 window_id: int = -1,  # default value
                 fps: int = 10):
        """
        Initialize all necessary parameters and generate a basic GUI layout
        that can then be augmented using `self.augment_layout`.
        :param parent: A wx.Frame parent (often Null). If it is non-Null,
            the frame will be minimized when its parent is minimized and
            restored when it is restored.
        :param window_id: The window identifier.
        :param title: The caption to be displayed on the frame's title bar.
        :param capture: Original video source to get the frames from.
        :param fps: Frames per second at which to display camera feed.
        """
        # Make sure the capture device could be set up
        self.capture = capture
        success, frame = self._acquire_frame()
        if not success:
            print("Could not acquire frame from camera.")
            raise SystemExit()
        self.imgHeight, self.imgWidth = frame.shape[:2]

        super().__init__(parent, window_id, title,
                         size=(self.imgWidth, self.imgHeight + 20))
        self.fps = fps
        self.bmp = wx.Bitmap.FromBuffer(self.imgWidth, self.imgHeight, frame)

        # set up periodic screen capture
        self.timer = wx.Timer(self)
        self.timer.Start(1000. / self.fps)
        self.Bind(wx.EVT_TIMER, self._on_next_frame)

        # set up video stream
        self.video_pnl = wx.Panel(self, size=(self.imgWidth, self.imgHeight))
        self.video_pnl.SetBackgroundColour(wx.BLACK)
        self.video_pnl.Bind(wx.EVT_PAINT, self._on_paint)

        # display the button layout beneath the video stream
        self.panels_vertical = wx.BoxSizer(wx.VERTICAL)
        self.panels_vertical.Add(self.video_pnl, 1, flag=wx.EXPAND | wx.TOP,
                                 border=1)

        self.augment_layout()

        # round off the layout by expanding and centering
        self.SetMinSize((self.imgWidth, self.imgHeight))
        self.SetSizer(self.panels_vertical)
        self.Centre()

    def augment_layout(self):
        """ Augment custom layout elements to the GUI.
        This method is called in the class constructor, after initializing
        common parameters. Every GUI contains the camera feed in the variable
        `self.video_pnl`. Additional layout elements can be added below
        the camera feed by means of the method `self.panels_vertical.Add`
        """
        raise NotImplementedError()

    def _on_next_frame(self, event):
        """
        Capture a new frame from the capture device,
        send an RGB version to `self.process_frame`, refresh.
        """
        success, frame = self._acquire_frame()
        if success:
            # process current frame
            frame = self.process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # update buffer and paint (EVT_PAINT triggered by Refresh)
            self.bmp.CopyFromBuffer(frame)
            self.Refresh(eraseBackground=False)

    def _on_paint(self, event):
        """ Draw the camera frame stored in `self.bmp` onto `self.video_pnl`.
        """
        wx.BufferedPaintDC(self.video_pnl).DrawBitmap(self.bmp, 0, 0)

    def _acquire_frame(self) -> (bool, np.ndarray):
        """ Capture a new frame from the input device
        :return: (success, frame)
            Whether acquiring was successful and current frame.
        """
        return self.capture.read()

    def process_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Process the frame of the camera (or other capture device)
        :param frame_rgb: Image to process in rgb format, of shape (H, W, 3)
        :return: Processed image in rgb format, of shape (H, W, 3)
        """
        raise NotImplementedError()


# ------------------------------------------------------------------------------------------------------
# class CameraCalibration
# ------------------------------------------------------------------------------------------------------

class CameraCalibration(BaseLayout):
    """Camera calibration

        Performs camera calibration on a webcam video feed using
        the chessboard approach described here:
        http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    """

    def augment_layout(self):
        pnl = wx.Panel(self, -1)
        self.button_calibrate = wx.Button(pnl, label='Calibrate Camera')
        self.Bind(wx.EVT_BUTTON, self._on_button_calibrate)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.button_calibrate)
        pnl.SetSizer(hbox)

        self.panels_vertical.Add(pnl, flag=wx.EXPAND | wx.BOTTOM | wx.TOP,
                                 border=1)

        # setting chessboard size (size of grid - 1)
        # (7,7) for the standard chessboard
        self.chessboard_size = (7, 7)

        # prepare object points
        self.objp = np.zeros((np.prod(self.chessboard_size), 3),
                             dtype=np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                                    0:self.chessboard_size[1]].T.reshape(-1, 2)

        # prepare recording
        self.recording = False
        self.record_min_num_frames = 15
        self._reset_recording()

    def process_frame(self, frame):
        """Processes each frame

            If recording mode is on (self.recording==True), this method will
            perform all the hard work of the camera calibration process:
            - for every frame, until enough frames have been processed:
                - find the chessboard corners
                - refine the coordinates of the detected corners
            - after enough frames have been processed:
                - estimate the intrinsic camera matrix and distortion
                  coefficients

            :param frame: current RGB video frame
            :returns: annotated video frame showing detected chessboard corners
        """
        # if we are not recording, just display the frame
        if not self.recording:
            return frame

        # else we're recording
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        if self.record_cnt < self.record_min_num_frames:
            # need at least some number of chessboard samples before we can
            # calculate the intrinsic matrix

            ret, corners = cv2.findChessboardCorners(img_gray,
                                                     self.chessboard_size,
                                                     None)
            if ret:
                print(f"{self.record_min_num_frames - self.record_cnt} chessboards remain")
                cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)

                # refine found corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30, 0.01)
                cv2.cornerSubPix(img_gray, corners, (9, 9), (-1, -1), criteria)

                self.obj_points.append(self.objp)
                self.img_points.append(corners)
                self.record_cnt += 1

        else:
            # we have already collected enough frames, so now we want to
            # calculate the intrinsic camera matrix (K) and the distortion
            # vector (dist)
            print("Calibrating...")
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points,
                                                             self.img_points,
                                                             (self.imgHeight,
                                                              self.imgWidth),
                                                             None, None)
            print("K=", K)
            print("dist=", dist)

            # double-check reconstruction error (should be as close to zero as
            # possible)
            mean_error = 0
            for obj_point, rvec, tvec, img_point in zip(
                    self.obj_points, rvecs, tvecs, self.img_points):
                img_points2, _ = cv2.projectPoints(
                    obj_point, rvec, tvec, K, dist)
                error = cv2.norm(img_point, img_points2,
                                 cv2.NORM_L2) / len(img_points2)
                mean_error += error

            print("mean error=", mean_error)

            self.recording = False
            self._reset_recording()
            self.button_calibrate.Enable()
        return frame

    def _on_button_calibrate(self, event):
        """Enable recording mode upon pushing the button"""
        self.button_calibrate.Disable()
        self.recording = True
        self._reset_recording()

    def _reset_recording(self):
        """Disable recording mode and reset data structures"""
        self.record_cnt = 0
        self.obj_points = []
        self.img_points = []


# ------------------------------------------------------------------------------------------------------
# 3D Scene Reconstruction using structure from motion:  Trial by sampe images
# ------------------------------------------------------------------------------------------------------

img_path = 'C:\\Users\\kosei-wada\\Desktop\\reference\\3DSceneReconstructionUsingStructureFromMotion\\sample_images'


"""
OpenCV with Python Blueprints
Chapter 4: 3D Scene Reconstruction Using Structure From Motion

An app to detect and extract structure from motion on a pair of images
using stereo vision. We will assume that the two images have been taken
with the same camera, of which we know the internal camera parameters. If
these parameters are not known, use calibrate.py to estimate them.

The result is a point cloud that shows the 3D real-world coordinates
of points in the scene.
"""

# camera matrix and distortion coefficients
# can be recovered with calibrate.py
# but the examples used here are already undistorted, taken with a camera
# of known K
K = np.array([[2759.48 / 4, 0, 1520.69 / 4, 0, 2764.16 / 4,
               1006.81 / 4, 0, 0, 1]]).reshape(3, 3)
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

print(K)
print(d)

scene = SceneReconstruction3D(K, d)

# load a pair of images for which to perform SfM
img1 = os.path.join(img_path, '0004.png')
img2 = os.path.join(img_path, '0005.png')

scene.load_image_pair(img1, img2)

# draw 3D point cloud of fountain
# use "pan axes" button in pyplot to inspect the cloud (rotate and zoom
# to convince you of the result)
# scene.draw_epipolar_lines()
# scene.plot_rectified_images()
scene.plot_optic_flow()
scene.plot_point_cloud()


# ------------------------------------------------------------------------------------------------------
# 3D Scene Reconstruction using structure from motion for video frame
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

rate = 4
# rate = 1


# capture = cv2.VideoCapture(0)
# assert capture.isOpened(), "Can not connect to camera"
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
# # start graphical user interface
# app = wx.App()
# layout = CameraCalibration(capture, title='Camera Calibration', fps=2)
# layout.Show(True)
# app.MainLoop()


# ----------
# imgA
vid = cv2.VideoCapture(video_file3)
# start_frame = 1
# end_frame = start_frame + 10
start_frame = 10000
end_frame = start_frame + 80

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

# vid2 = cv2.VideoCapture(video_file4)
# # start_frame2 = 3100
# # end_frame2 = start_frame2 + 1
# start_frame2 = 2050
# end_frame2 = start_frame2 + 1

vid = cv2.VideoCapture(video_file3)
start_frame2 = 10000
end_frame2 = start_frame2 + 82

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

frame2 = img_proc(frame2)
frame3 = img_proc(frame3)

cv2.imwrite(os.path.join(base_path, 'tmp2.png'), frame2)
cv2.imwrite(os.path.join(base_path, 'tmp3.png'), frame3)

cv2.imshow("frame2", frame2)
cv2.imshow("frame3", frame3)


# ----------
# K = np.array([[2759.48 / 4, 0, 1520.69 / 4, 0, 2764.16 / 4,
#                1006.81 / 4, 0, 0, 1]]).reshape(3, 3)
# d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

K = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]).reshape(3, 3)
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

print(K)
print(d)

scene = SceneReconstruction3D(K, d)

# load a pair of images for which to perform SfM
img2 = os.path.join(os.path.join(base_path, 'tmp2.png'))
img3 = os.path.join(os.path.join(base_path, 'tmp3.png'))

scene.load_image_pair(img2, img3)

# draw 3D point cloud of fountain
# use "pan axes" button in pyplot to inspect the cloud (rotate and zoom
# to convince you of the result)
# scene.draw_epipolar_lines()
# scene.plot_rectified_images()
scene.plot_optic_flow()

scene.plot_point_cloud()


