
import numpy as np
import cv2
import imutils


# ------------------------------------------------------------------------------------------------------
# basic setting
# ------------------------------------------------------------------------------------------------------

base_path = 'C:\\Users\\kosei-wada\\Desktop\\hand_trajectory_analysis'

# video_file = os.path.join(base_path, '01_video\\Gamma_Assembly_192_168_32_69_1_20210223_60min_result.mp4')

video_file = os.path.join(base_path, 'originals\\video\\PCDGammaAssembly\\Gamma_Assembly_192_168_32_69_1_20210223142400_10min.mp4')


# ----------
vid = cv2.VideoCapture(video_file)

print(vid.isOpened())

frame_count = int(vid.get(7))
frame_rate = vid.get(5)
frame_sec = (1.0 / frame_rate)

print(frame_count)
print(frame_rate)
print(frame_sec)


# ------------------------------------------------------------------------------------------------------
# load data
# ------------------------------------------------------------------------------------------------------

df1 = pd.read_csv(MART_DF1)

dfstr1 = pd.read_csv(MART_DFSTR1)



########################################################################################################
# ------------------------------------------------------------------------------------------------------
# define area  +  detect color change
# ------------------------------------------------------------------------------------------------------

def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(max(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


start_frame = 1000
vid.set(1, start_frame)

rate = 1

i = 0

# ----------
while True:

    is_read, frame = vid.read()

    if not (is_read):
        break

    i += 1
    frame = cv2.resize(frame, (frame.shape[1]*rate, frame.shape[0]*rate))

    clone = frame.copy()
    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # edge = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)
    wide = cv2.Canny(gray, 10, 50)
    # tight = cv2.Canny(gray, 225, 250)
    # auto = auto_canny(gray)
    # sob = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)

    # cv2.imshow('4 sobel', sob)
    # cv2.imshow('3 canny', np.hstack([wide, tight, auto]))
    # cv2.imshow('2 Laplacian', edge)
    # cv2.imshow('1 gray', gray)
    cv2.imshow('original', frame)

    cv2.imshow('canny', wide)
    # cv2.imshow('auto canny', auto)

    if cv2.waitKey(3) & 0xFF == ord('q'):
            break

# ----------
vid.release()

cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------------------------
# check color transition
# ------------------------------------------------------------------------------------------------------

df = pd.DataFrame(list, columns={'rmean', 'gmean', 'bmean'})

df.plot()


# df['sec_orig'] = df1['sec_orig'][0:df.shape[0]]
# df['rmean'].plot(subplots=True, x='frames', figsize=(10, 5))[0]
# df['start_flag'] = (df['rmean'] >= 195) * 1
# df['start_flag'] = ((df['rmean'] >= 195) & (df['rmean'].diff(1).fillna(0) > 10)) * 1
# df['start_flag'].value_counts()
# df[df['start_flag'] == 1].sec_orig




