
import os
import cv2
# import pprint

# cv2 build information
print(cv2.getBuildInformation())


# ------------------------------------------------------------------------------------------
# webcam
# ------------------------------------------------------------------------------------------
base_path = '/home/kswada/kw/image_processing/video'

save_video_dir = os.path.join(base_path, 'video_output')


# ----------
stream = cv2.VideoCapture(0)


# ----------
width = stream.get(cv2.CAP_PROP_FRAME_WIDTH) 
height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT) 
fps = stream.get(cv2.CAP_PROP_FPS)


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
# available fourcc  (fourcc = -1 shows the list)
cv2.VideoWriter(
    filename='tmp.mp4',
    fourcc=-1,
    fps=fps,
    frameSize=(int(width), int(height)))

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')


# ----------
# change property
stream = cv2.VideoCapture(0)

# stream.set(cv2.CAP_PROP_AUTO_WB, 1.0)
# save_suffix = 'autowb_on'

# stream.set(cv2.CAP_PROP_AUTO_WB, 0.0)
# save_suffix = 'autowb_off'

stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
save_suffix = 'autoexposure_0'


# ------------
video_output_path = os.path.join(save_video_dir, f'experiment_{save_suffix}.mp4')

writer = cv2.VideoWriter(
    filename=video_output_path,
    fourcc=fourcc,
    fps=fps,
    frameSize=(int(width), int(height)))


# ----------
for _ in range(30*10):
# while True:
    ret_val, frame = stream.read()
    if not ret_val:
        break
    cv2.imshow('video', frame)
    writer.write(frame)
    # ----------
    # 0xff == 27 (escape key)
    if cv2.waitKey(5) & 0xFF == 27:
      break


# ----------
stream.release()
writer.release()

cv2.destroyAllWindows()




