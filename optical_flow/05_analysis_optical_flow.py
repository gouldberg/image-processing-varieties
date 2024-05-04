
import numpy as np
import cv2
import imutils

from scipy.fft import rfft, rfftfreq
from scipy import signal
from scipy import fftpack

from sklearn.linear_model import LinearRegression
from patsy import cr
import csaps

from spectrum import aryule, arma2psd

from collections import Counter

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
# tracking key points
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

# video = cv2.VideoCapture(video_file3)
#
# start_frame = 11550
# end_frame = start_frame + 800


def img_proc(img):
    kernel = np.ones((3, 3), np.uint8)
    # img_obj = adjust_gamma(img, gamma=0.8)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (3, 3), 0)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj


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

xmin = 80
xmax = 280
ymin = 0
ymax = 140


# ----------
video = cv2.VideoCapture(video_file)

# start_frame = screw_time_list[0][0]
start_frame = 0
end_frame = screw_time_list[0][1]


# ----------
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
ret, prev_frame = video.read()

prev_frame = cv2.resize(prev_frame, (prev_frame.shape[1] * rate, prev_frame.shape[0] * rate))
# only screwing image
prev_frame2 = prev_frame[ymin:ymax, xmin:xmax]
prev_gray = img_proc(prev_frame2)
# prev_gray = cv2.cvtColor(prev_frame2, cv2.COLOR_RGB2GRAY)

# feature_params = {
#     "maxCorners": 500,  # 特徴点の上限数
#     "qualityLevel": 0.6,  # 閾値　（高いほど特徴点数は減る)
#     "minDistance": 8,  # 特徴点間の距離 (近すぎる点は除外)
#     "blockSize": 8
# }

feature_params = {
    "maxCorners": 50,  # 特徴点の上限数
    "qualityLevel": 0.4,  # 閾値　（高いほど特徴点数は減る)
    "minDistance": 10,  # 特徴点間の距離 (近すぎる点は除外)
    "blockSize": 8
}

lk_params = {
    "winSize": (10, 10),  # 特徴点の計算に使う周辺領域サイズ
    # "winSize": (40, 40),  # 特徴点の計算に使う周辺領域サイズ
    "maxLevel": 2,  # ピラミッド数 (デフォルト0で、2の場合は1/4の画像まで使われる)
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # 探索アルゴリズムの終了条件
}


p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, prev_gray, p0, None, **lk_params)


for p in p0:
    x, y = p.ravel()
    cv2.circle(prev_frame2, (x, y), 3, (0, 255, 255), -1)

cv2.imshow("image1", prev_frame)
cv2.imshow("image2", prev_frame2)
cv2.imshow("image3", prev_gray)
cv2.destroyAllWindows()


# ----------

color = np.random.randint(0, 255, (500, 3))

vid = cv2.VideoCapture(video_file)

# start_frame2 = screw_time_list[j][0]
# end_frame2 = screw_time_list[j][1]

start_frame2 = 0
end_frame2 = 3599*30

vid.set(1, start_frame2)

mask = np.zeros_like(prev_frame2)

for i in range(start_frame2, end_frame2, 1):
    print(f'{i} / {end_frame2}')

    ret, frame = vid.read()
    if not (ret):
        break
    # ret, frame = vid.read()
    # if not (ret):
    #     break

    frame = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
    frame = frame[ymin:ymax, xmin:xmax]
    frame_gray = img_proc(frame)
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
    # print(p1)

    if p1 is not None:
        identical_p1 = p1[status == 1]
        identical_p0 = p0[status == 1]

        for k, (p1, p0) in enumerate(zip(identical_p1, identical_p0)):
            p1_x, p1_y = p1.ravel()
            p0_x, p0_y = p0.ravel()
            mask = cv2.line(mask, (p1_x, p1_y), (p0_x, p0_y), color[k].tolist(), 2)
            frame = cv2.circle(frame, (p1_x, p1_y), 5, color[k].tolist(), -1)

        image = cv2.add(frame, mask)
        cv2.imshow("image4", image)
        prev_gray = frame_gray.copy()
        p0 = identical_p1.reshape(-1, 1, 2)
    else:
        cv2.imshow("image4", frame)
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)
        prev_gray = frame_gray.copy()

    if cv2.waitKey(3) & 0xFF == ord('q'):
            break


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# tracking key points
# ------------------------------------------------------------------------------------------------------

def img_proc(img):
    kernel = np.ones((3, 3), np.uint8)
    # img_obj = adjust_gamma(img, gamma=0.8)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (3, 3), 0)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def anorm2(a):
    return (a*a).sum(-1)


# 0419 - 0421
vid_original_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video\\originals'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(vid_original_path, vid_file)

rate = 1
fps = 30

xmin = 80
xmax = 280
ymin = 0
ymax = 140

color = np.random.randint(0, 255, (500, 3))

# lk_params = dict( winSize  = (15, 15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
# feature_params = dict( maxCorners = 3,
#                        qualityLevel = 0.5,
#                        minDistance = 10,
#                        blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 3,
                       qualityLevel = 0.5,
                       minDistance = 50,
                       blockSize = 7 )


cam = cv2.VideoCapture(video_file)
track_len = 128
detect_interval = 1
tracks = []
tracks_sec_id = []
tracks_all = []
# frame_idx = 6000
frame_idx = 0

cam.set(1, frame_idx)
frame_count = 3599 * 30
trackid = -1

while True:
    print(f'{frame_idx} / {frame_count}')
    _ret, frame = cam.read()
    if not (_ret):
        break
    _ret, frame = cam.read()
    if not (_ret):
        break

    frame = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
    frame = frame[ymin:ymax, xmin:xmax]
    frame_gray = img_proc(frame)
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 2
        new_tracks = []
        new_tracks_sec_id = []
        for tr, tr_sec_id, (x, y), good_flag in zip(tracks, tracks_sec_id, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            tr_sec_id.append((frame_idx/fps, tr_sec_id[0][1]))
            tracks_all.append([frame_idx/fps, tr_sec_id[0][1], x, y])
            if len(tr) > track_len:
                del tr[0]
                del tr_sec_id[0]
            new_tracks.append(tr)
            new_tracks_sec_id.append(tr_sec_id)
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        tracks = new_tracks
        tracks_sec_id = new_tracks_sec_id
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        draw_str(vis, (20, 20), 'track count: %d' % len(tracks))

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                trackid += 1
                tracks.append([(x, y)])
                tracks_sec_id.append([(frame_idx / fps, trackid)])

    frame_idx += 2
    prev_gray = frame_gray
    cv2.imshow('lk_track', vis)

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break


# sec_orig = [tracks_all[i][0] for i in range(len(tracks_all))]
# trackid = [tracks_all[i][1] for i in range(len(tracks_all))]
# x = [tracks_all[i][2] for i in range(len(tracks_all))]
# y = [tracks_all[i][3] for i in range(len(tracks_all))]
#
# track_all_df = pd.DataFrame({'sec_orig': sec_orig, 'trackid': trackid, 'x': x, 'y': y}).sort_values(['trackid','sec_orig'])
# track_all_df.reset_index(drop=True,inplace=True)
#
# track_all_df.describe()
#
# track_all_df.to_csv(os.path.join(base_path, '03_datamart\\optflow.csv'), index=False)
# print(len(track_all_df))
#
#
# # ----------
# thresh_len = 2**5
#
# track_all_df2 = track_all_df.assign(cnt=1)
# tmp = track_all_df2[['trackid','cnt']].groupby(['trackid']).agg('sum').reset_index()
# tmp_trackid = tmp[tmp.cnt >= thresh_len].trackid
# track_all_df2 = track_all_df2[track_all_df2.trackid.isin(tmp_trackid)]
# track_all_df2 = track_all_df2[['sec_orig','trackid','x','y']]
#
# track_all_df2.to_csv(os.path.join(base_path, '03_datamart\\optflow_small.csv'), index=False)
# print(len(track_all_df2))


# ----------
for trackid in range(10):
    tmp = track_all_df[track_all_df.trackid == trackid]
    pd.DataFrame(tmp.x).plot()

pd.DataFrame(tracks[0]).plot()


# ----------
lst = []

len_track = len(track_all_df.trackid.unique())

thresh_val = 1.0

i = 0
for trackid in track_all_df.trackid.unique():

    print(f'{i} / {len_track} - {trackid}')
    datx = track_all_df[track_all_df.trackid == trackid].x.values
    daty = track_all_df[track_all_df.trackid == trackid].y.values

    if len(datx) >= 100:
        sp_x = csaps.csaps(list(range(len(datx))), datx, list(range(len(datx))), smooth=0.005)
        datx_res = datx - sp_x

        sp_y = csaps.csaps(list(range(len(daty))), daty, list(range(len(daty))), smooth=0.005)
        daty_res = daty - sp_y

        # pd.DataFrame(datx_res).plot()
        # pd.DataFrame(datx).plot()
        # pd.DataFrame(datx_res).plot()
        # pd.DataFrame(daty).plot()

        # n = len(datx)
        # # note that here we get 1 frame by 2 frames
        # dt = (1*2)/fps

        # fx = fftpack.rfft(datx_res)/(n/2)
        # fy = fftpack.rfft(daty_res)/(n/2)
        # freq = fftpack.fftfreq(n, dt)

        # fx = fftpack.rfft(datx)/(n/2)
        # fy = fftpack.rfft(daty)/(n/2)
        # freq = fftpack.fftfreq(n, dt)

        # maxvalx = max(np.abs(fx[1:int(n/2)]))
        # maxvaly = max(np.abs(fy[1:int(n/2)]))


        # ----------
        Fs = int(fps * 0.5)
        LEN = len(datx)
        win = signal.get_window('hamming', LEN)
        enbw = LEN * np.sum(win ** 2) / np.sum(win) ** 2
        rfft_datx = rfft(datx_res * win)
        rfft_daty = rfft(daty_res * win)
        rfft_freq = rfftfreq(LEN, d=1.0 / Fs)
        sp_rdatx = np.abs(rfft_datx) ** 2 / (Fs * LEN * enbw)
        sp_rdaty = np.abs(rfft_daty) ** 2 / (Fs * LEN * enbw)

        sp_rdatx[1:-1] *= 2
        sp_rdaty[1:-1] *= 2

        maxvalx = max(sp_rdatx)
        maxvaly = max(sp_rdaty)
        overcntx = sum(sp_rdatx >= thresh_val)
        overcnty = sum(sp_rdaty >= thresh_val)

    else:
        maxvalx, maxvaly, overcntx, overcnty = 0, 0, 0, 0

    lst.append([trackid, maxvalx, maxvaly, overcntx, overcnty])

    # if valx > thresh_amp:
    #     pd.DataFrame(np.abs(fx[1:int(n/2)])).plot()
    #
    # if valy > thresh_amp:
    #     pd.DataFrame(np.abs(fy[1:int(n/2)])).plot()

        # plt.plot(freq[1:int(n/2)], np.abs(fy[1:int(n/2)]))
        # plt.ylabel("Amplitude")
        # plt.xlabel("Frequency [Hz]")

    i += 1


trackinfo = pd.DataFrame(lst)
trackinfo.columns = ['trackid', 'maxvalx', 'maxvaly', 'overcntx', 'overcnty']

print(trackinfo)
print(trackinfo.describe())


# ----------
trackinfo.maxvalx.hist(bins=20)

objtrack1 = trackinfo[(trackinfo.maxvalx > thresh_val)&(trackinfo.overcntx == 1)].trackid.values
objtrack2 = trackinfo[(trackinfo.maxvaly > thresh_val)&(trackinfo.overcnty == 1)].trackid.values

print(objtrack1)
print(objtrack2)
print(len(objtrack1))
print(len(objtrack2))


for trackid in objtrack1[40:60]:
    tmp = track_all_df[track_all_df.trackid == trackid]
    pd.DataFrame(tmp.x).plot()


for trackid in objtrack2:
    tmp = track_all_df[track_all_df.trackid == trackid]
    pd.DataFrame(tmp.y).plot()


# ----------
# Power Spectral Density:

# trackid = 10270

# trackid = 7528
# trackid = 20643
# trackid = 23794

trackid = 14130
trackid = 14151
trackid = 15814

trackid = 15821
trackid = 20653
trackid = 33816

datx = track_all_df[track_all_df.trackid == trackid].x.values
daty = track_all_df[track_all_df.trackid == trackid].y.values

sp_x = csaps.csaps(list(range(len(datx))), datx, list(range(len(datx))), smooth=0.005)
datx_res = datx - sp_x

sp_y = csaps.csaps(list(range(len(daty))), daty, list(range(len(daty))), smooth=0.005)
daty_res = daty - sp_y

# here 1 get from 2 frames
Fs = int(fps * 0.5)

LEN = len(datx)

win = signal.get_window('hamming', LEN)

enbw = LEN * np.sum(win**2) / np.sum(win)**2

rfft_datx = rfft(datx_res * win)
rfft_daty = rfft(daty_res * win)

rfft_freq = rfftfreq(LEN, d = 1.0/Fs)

sp_rdatx = np.abs(rfft_datx)**2 / (Fs * LEN * enbw)
sp_rdaty = np.abs(rfft_daty)**2 / (Fs * LEN * enbw)

sp_rdatx[1:-1] *= 2
sp_rdaty[1:-1] *= 2

print(sum(sp_rdatx >= thresh_val))
print(sum(sp_rdaty >= thresh_val))

# plt.figure(figsize=[7,4])
plt.figure(211)
plt.plot(rfft_freq, sp_rdatx)
plt.plot(rfft_freq, sp_rdaty)
plt.xlabel('Frequency')

pd.DataFrame(datx_res).plot()
pd.DataFrame(datx).plot()


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Maximum Entropy Method
# ------------------------------------------------------------------------------------------------------

track_all_df = pd.read_csv(os.path.join(base_path, '03_datamart\\optflow.csv'))
print(len(track_all_df))

thresh_len = 2**4

track_all_df2 = track_all_df.assign(cnt=1)
tmp = track_all_df2[['trackid','cnt']].groupby(['trackid']).agg('sum').reset_index()
tmp_trackid = tmp[tmp.cnt >= thresh_len].trackid
track_all_df2 = track_all_df2[track_all_df2.trackid.isin(tmp_trackid)]
track_all_df2 = track_all_df2[['sec_orig','trackid','x','y']]

track_all_df2.to_csv(os.path.join(base_path, '03_datamart\\optflow_small.csv'), index=False)
print(len(track_all_df2))


# ----------
track_all_df = pd.read_csv(os.path.join(base_path, '03_datamart\\optflow_small.csv'))
print(len(track_all_df))
len(track_all_df.trackid.unique())


lst = []
len_track = len(track_all_df.trackid.unique())
thresh_val = 1.0
Fsr = 30

i = 0
for trackid in track_all_df.trackid.unique():

    print(f'{i} / {len_track} - {trackid}')
    datx = track_all_df[track_all_df.trackid == trackid].x.values
    daty = track_all_df[track_all_df.trackid == trackid].y.values

    if len(datx) >= 100:
        sp_x = csaps.csaps(list(range(len(datx))), datx, list(range(len(datx))), smooth=0.005)
        datx_res = datx - sp_x

        sp_y = csaps.csaps(list(range(len(daty))), daty, list(range(len(daty))), smooth=0.005)
        daty_res = daty - sp_y

        N = len(datx)
        win = signal.get_window('hanning', N)

        ARx, Px, kx = aryule(datx_res * win, N // 8)
        ARy, Py, ky = aryule(daty_res * win, N // 8)
        PSDx = arma2psd(ARx, NFFT=N)
        PSDy = arma2psd(ARy, NFFT=N)
        # fr = np.linspace(0, Fsr, N, endpoint=False)

        maxvalx = max(PSDx[0:int(Fsr/2)])
        maxvaly = max(PSDy[0:int(Fsr/2)])
        overcntx = sum(PSDx[0:int(Fsr/2)] >= thresh_val)
        overcnty = sum(PSDy[0:int(Fsr/2)] >= thresh_val)

    else:
        maxvalx, maxvaly, overcntx, overcnty = 0, 0, 0, 0

    lst.append([trackid, maxvalx, maxvaly, overcntx, overcnty])
    i += 1


trackinfo = pd.DataFrame(lst)
trackinfo.columns = ['trackid', 'maxvalx', 'maxvaly', 'overcntx', 'overcnty']

print(trackinfo)
print(trackinfo.describe())

trackinfo.maxvalx.hist(bins=20)

objtrack1 = trackinfo[(trackinfo.maxvalx > thresh_val)&(trackinfo.overcntx == 1)].trackid.values
objtrack2 = trackinfo[(trackinfo.maxvaly > thresh_val)&(trackinfo.overcnty == 1)].trackid.values

print(objtrack1)
print(objtrack2)
print(len(objtrack1))
print(len(objtrack2))


for trackid in objtrack1[40:60]:
    tmp = track_all_df[track_all_df.trackid == trackid]
    pd.DataFrame(tmp.x).plot()


for trackid in objtrack2:
    tmp = track_all_df[track_all_df.trackid == trackid]
    pd.DataFrame(tmp.y).plot()



########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Dense Optical Flow
# ------------------------------------------------------------------------------------------------------

# 0419 - 0421
vid_original_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video\\originals'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(vid_original_path, vid_file)

rate = 1
fps = 30

xmin = 80
xmax = 280
ymin = 0
ymax = 140

cap = cv2.VideoCapture(video_file)

frame_idx = 0

cap.set(1, frame_idx)

frame_count = 3599 * 30


ret, frame1 = cap.read()
# frame1 = cv2.resize(frame1, (frame1.shape[1] * rate, frame1.shape[0] * rate))
# frame1 = frame1[ymin:ymax, xmin:xmax]

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    # frame2 = cv2.resize(frame2, (frame2.shape[1] * rate, frame2.shape[0] * rate))
    # frame2 = frame2[ymin:ymax, xmin:xmax]

    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',frame2)
    cv2.imshow('rgb',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite(os.path.join(base_path, '03_datamart\\opticalfb.png'), frame2)
        cv2.imwrite(os.path.join(base_path, '03_datamart\\opticalhsv.png'), rgb)
    prvs = next

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Dense Optical Flow - 2
# ------------------------------------------------------------------------------------------------------

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    fx, fy = flow[y.astype('uint32'), x.astype('uint32')].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


# ----------
# print(len(flow))
# flow[0]
#
# img = gray
# step = 16
# h, w = img.shape[:2]
# y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
# fx, fy = flow[y.astype('uint32'), x.astype('uint32')].T
# lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
# lines = np.int32(lines + 0.5)
# vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cv2.polylines(vis, lines, 0, (0, 255, 0))
# for (x1, y1), (x2, y2) in lines:
#     cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

h, w = flow.shape[:2]
fx, fy = flow[:, :, 0], flow[:, :, 1]

ang = np.arctan2(fy, fx) + np.pi
ang = np.arctan2(fy, fx)
# ang = ang * (180 / np.pi / 2)
ang = ang * (180 / np.pi)


v = np.sqrt(fx * fx + fy * fy)

hsv = np.zeros((h, w, 3), np.uint8)
hsv[..., 0] = ang
hsv[..., 1] = 255
hsv[..., 2] = np.minimum(v * 4, 255)


pd.DataFrame(v).describe()
pd.DataFrame(ang).describe()

v_hist = []
for i in range(len(v)):
    v_hist.append(np.histogram(v[i],bins=np.linspace(0, 5, 9))[0])

fx_hist = []
for i in range(len(fx)):
    fx_hist.append(np.histogram(fx[i],bins=np.linspace(0, 5, 9))[0])

fy_hist = []
for i in range(len(fy)):
    fy_hist.append(np.histogram(fy[i],bins=np.linspace(0, 5, 9))[0])

ang_hist = []
for i in range(len(ang)):
    ang_hist.append(np.histogram(ang[i],bins=np.linspace(-180, 180, 9))[0])


# ----------
# 0419 - 0421
vid_original_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video\\originals'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(vid_original_path, vid_file)

rate = 1
fps = 30

xmin = 80
xmax = 280
ymin = 0
ymax = 140

frame_idx = 6200
# frame_idx = 1

frame_count = 3599 * 30

cam = cv2.VideoCapture(video_file)
cam.set(1, frame_idx)

ret, prev = cam.read()
prev = cv2.resize(prev, (int(prev.shape[1] * rate), int(prev.shape[0] * rate)))
prev = prev[ymin:ymax, xmin:xmax]
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

show_hsv = True
show_glitch = False
cur_glitch = prev.copy()

st = time.time()
while frame_idx <= frame_count:
    ret, img = cam.read()
    if not ret:
        break

    ret, img = cam.read()
    if not ret:
        break

    frame_idx += 2
    img = cv2.resize(img, (int(img.shape[1] * rate), int(img.shape[0] * rate)))
    img = img[ymin:ymax, xmin:xmax]

    # print(f'{frame_idx}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # prevgray: first 8-bit single-channel input image
    # next: second input image of the same size and the same type as prev
    # None: computed flow image that has the same size as prev and type CV_32FC2
    # pyr_scale = 0.5:  parameter, specifying the image scale (<1) to build pyramids for each image;
    #                   pyr_scale = 0.5 means a classical pyramid, where each next layer is twice smaller than the previous one
    # levels = 3:  number of pyramid layers including the initial image
    #  - levels = 1 means that no extra layers are created and only the original images are used
    # winsize = 15:  averaging window size, larger values increase the algorithm robustness to image noise
    #                and give more chances for fast motion detection, but yield more blurred motion field
    # iterations = 3:  number of iterations the algorithm does at each pyramid level
    # poly_n = 5:  size of the pixel neighborhood used to find polynomial expansion in each pixel
    #             larger values mean that the image will be approximated with smoother surfaces,
    #             yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7
    # poly_sigma = 1.2:  standard deviation of the Gaussian that is used to smooth derivatives
    #              used as a basis for the polynomial expansion
    #              for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
    # flags = 0:
    #   - OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
    #   - OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize * winsize filter instead of a box filter of the same size
    #     for optical flow estimation
    #     usually this option gives z more accurate flow than with a box filter, at the cost of lower speed;
    #     normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.

    flow = cv2.calcOpticalFlowFarneback(prev = prevgray,
                                        next = gray,
                                        flow = None,
                                        pyr_scale = 0.5,
                                        levels = 3,
                                        winsize = 15,
                                        iterations = 3,
                                        poly_n = 5,
                                        poly_sigma = 1.2,
                                        flags = 0)

    prevgray = gray
    cv2.imshow('flow', draw_flow(gray, flow))
    # cv2.waitKey(1500)

    if show_hsv:
        cv2.imshow('flow HSV', draw_hsv(flow))
    # if show_glitch:
    #     cur_glitch = warp_flow(cur_glitch, flow)
    #     cv2.imshow('glitch', cur_glitch)

    # ch = 0xFF & cv2.waitKey(5)
    # print(f'{ch}')
    #
    # if ch == 27:
    #     break
    # if ch == ord('1'):
    #     show_hsv = not show_hsv
    #     print('HSV flow visualization is', ['off', 'on'][show_hsv])
    # if ch == ord('2'):
    #     show_glitch = not show_glitch
    #     if show_glitch:
    #         cur_glitch = img.copy()
    #     print('glitch is', ['off', 'on'][show_glitch])
    #
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

ed = time.time()

print(f'{ed - st: .2f} secs')

cv2.destroyAllWindows()


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Camshift
# ------------------------------------------------------------------------------------------------------

# 0419 - 0421
vid_original_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis\\01_video\\originals'
vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
video_file = os.path.join(vid_original_path, vid_file)

rate = 1
fps = 30

xmin = 80
xmax = 280
ymin = 0
ymax = 140

cap = cv2.VideoCapture(video_file)

ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 28,30,200,30
track_window = (c,r,w,h)
roi = frame[r:r+h, c:c+w]

frame = frame[ymin:ymax, xmin:xmax]

# set up the ROI for tracking
cv2.imshow('frame', frame)
cv2.imshow('roi', roi)


hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):

    print(f'{frame_idx} / {frame_count}')
    _ret, frame = cam.read()
    if not (_ret):
        break

    frame = frame[ymin:ymax, xmin:xmax]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # apply meanshift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame,[pts],True, 255,2)
    cv2.imshow('img2',img2)

    # k = cv2.waitKey(60) & 0xff
    # if k == 27:
    #     break
    # else:
    #     cv2.imwrite(chr(k)+".jpg",img2)

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


########################################################################################################
# ------------------------------------------------------------------------------------------------------
# Fisher Vector
# ------------------------------------------------------------------------------------------------------

import numpy as np

from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture as GMM


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


# ----------
K = 64
N = 1000

xx, _ = make_classification(n_samples=N)
xx_tr, xx_te = xx[: -100], xx[-100: ]
xx_tr.shape

gmm = GMM(n_components=K, covariance_type='diag')
gmm.fit(xx_tr)

fv = fisher_vector(xx_te, gmm)


gmm.fit(xx)