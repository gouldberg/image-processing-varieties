
import os
import cv2
# import pprint

# cv2 build information
print(cv2.getBuildInformation())


# ------------------------------------------------------------------------------------------
# video property
# ------------------------------------------------------------------------------------------

base_path = '/home/kswada/kw/image_processing/video'

video_dir = os.path.join(base_path, 'sample_videos')

video_path = os.path.join(video_dir, 'jurassic_park_intro.mp4')


# ----------
cap = cv2.VideoCapture(video_path)


print(f'opened: {cap.isOpened()}')


# ----------
# cv2 cap property
print(cv2.CAP_PROP_FRAME_WIDTH)


# ----------
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

secs = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

print(f'width: {width}   height: {height}')

print(f'fps: {fps}   frame_count: {frame_count}  secs: {secs}')

print(f'current frame: {cur_frame}')


# ----------
# update property

# set current frame
print(cap.set(cv2.CAP_PROP_POS_FRAMES, 100))



# ------------------------------------------------------------------------------------------
# video read and show
# ------------------------------------------------------------------------------------------

cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret_val, frame = cap.read()
    if not ret_val:
        break
    cv2.imshow('video', frame)
    if cv2.waitKey(5) & 0xFF == 27:
      break
    

# ----------
cv2.destroyAllWindows()

cap.release()


# ------------------------------------------------------------------------------------------
# webcam
# ------------------------------------------------------------------------------------------

stream = cv2.VideoCapture(0)


# ----------
# get property
width = stream.get(cv2.CAP_PROP_FRAME_WIDTH) 
height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT) 
fps = stream.get(cv2.CAP_PROP_FPS)
frame_count = stream.get(cv2.CAP_PROP_FRAME_COUNT)

print(f'width: {width}   height: {height}')
print(f'fps: {fps}   frame_count: {frame_count}')


# ----------
# exposure: 3.0
print(stream.get(cv2.CAP_PROP_AUTO_EXPOSURE))

# sharpness: 128
print(stream.get(cv2.CAP_PROP_SHARPNESS))

# focus: 0.0
print(stream.get(cv2.CAP_PROP_FOCUS))

# temperature: 4000.0
print(stream.get(cv2.CAP_PROP_TEMPERATURE))

# enable/ disable auto white-balance: 1.0
print(stream.get(cv2.CAP_PROP_AUTO_WB))

# white-balance color temperature: 4000
print(stream.get(cv2.CAP_PROP_WB_TEMPERATURE))

# -1
print(stream.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V))


# ----------
while True:
    ret_val, frame = stream.read()
    if not ret_val:
        break
    cv2.imshow('video', frame)
    # ----------
    # 0xff == 27 (escape key)
    if cv2.waitKey(5) & 0xFF == 27:
      break

stream.release()

cv2.destroyAllWindows()




