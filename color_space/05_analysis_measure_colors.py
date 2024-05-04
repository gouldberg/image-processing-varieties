
import numpy as np
import cv2
import imutils

import itertools


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


def img_proc_tmplt(img):
    kernel = np.ones((5, 5), np.uint8)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (7, 7), 0)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj

def img_proc_orig(img):
    kernel = np.ones((5, 5), np.uint8)
    img_obj = adjust_gamma(img, gamma=1.75)
    img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
    img_obj = cv2.GaussianBlur(img_obj, (7, 7), 0)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj


# -----------------------------------------------------------------------------------------------
# load data
# -----------------------------------------------------------------------------------------------

file_info = pd.read_csv(MART_FILE_INFO)
# df1 = pd.read_csv(MART_DF1)


# -----------
# champ_fname_list = list(file_info.fname[4:])
# nonchamp_fname_list1 = list(file_info.fname[:2])
# nonchamp_fname_list2 = list(file_info.fname[2:4])

champ_fname_list = list(file_info.fname[0:])
# nonchamp_fname_list1 = list(file_info.fname[:2])
# nonchamp_fname_list2 = list(file_info.fname[2:4])

# video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly'
video_orig_path = 'C:\\Users\\kosei-wada\\Desktop\hand_trajectory_analysis\\originals\\video\\PCDGammaAssembly\\pcd_0420'


# ----------
# time_slots = pd.read_csv(os.path.join(base_path, '03_datamart\\03_cycle_info.csv'))
time_slots = pd.read_csv(os.path.join(base_path, '01_data\\subcycle_info\\subcycle_info_PCDGamma_pcd0420_pm.txt'), sep='\t')


# -----------------------------------------------------------------------------------------------
# CHAMPION:  set grid
# -----------------------------------------------------------------------------------------------

width = 30

# objective area:  base
# Keyboard
xmin = 330
ymin = 55
xmax = xmin + width
ymax = ymin + width

# # GammaBase
xmin2 = 320
ymin2 = 170
xmax2 = xmin2 + width
ymax2 = ymin2 + width


# # -----------
# # generate grid to cover Keyboard and GammaBase and other area
# x = list(range(0, 640, 30))[1:-1][::2]
# y = [5] + list(range(-5, 480, 30))[2:-1][::2]
#
# grid_k0 = list(itertools.product(x,y))
#
# grids = []
# for i in range(len(grid_k0)):
#     grids.append((grid_k0[i][0], grid_k0[i][0]+30, grid_k0[i][1], grid_k0[i][1]+30))
#
# # add GammaBase
# grids.append((xmin, xmax, ymin, ymax))


# ----------
step = int(width / 2)
# add slightly adjacent area: keyboard
x2 = list(range(xmin-step*2, xmax+step*1, step))
y2 = list(range(ymin-step*2, ymax+step*1, step))

grid_k0 = list(itertools.product(x2,y2))

grid_k = []
for i in range(len(grid_k0)):
    grid_k.append((grid_k0[i][0], grid_k0[i][0]+width, grid_k0[i][1], grid_k0[i][1]+width))

# add slightly adjacent area: GammaBase
x3 = list(range(xmin2-step*2, xmax2+step*1, step))
y3 = list(range(ymin2-step*2, ymax2+step*1, step))

grid_g0 = list(itertools.product(x3,y3))

grid_g = []
for i in range(len(grid_g0)):
    grid_g.append((grid_g0[i][0], grid_g0[i][0]+width, grid_g0[i][1], grid_g0[i][1]+width))


# ----------
print(len(grid_k))
print(grid_k)

print(len(grid_g))
print(grid_g)


# -----------------------------------------------------------------------------------------------
# CHAMPION:  check the grid
# -----------------------------------------------------------------------------------------------

# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303165000_10min.mp4'
# start_frame = 2000
# end_frame = start_frame + 300

# start_frame = 5000
# end_frame = start_frame + 10

# something on work station
# start_frame = 2800
# end_frame = start_frame + 200


# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210223134400_10min.mp4'
# start_frame = 1
# end_frame = start_frame + 10
# start_frame = 3100
# end_frame = start_frame + 500

rate = 1

# vid_file = 'Gamma_Assembly_192_168_32_69_1_20201130180000_10min.mp4'

vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
# start_frame = 1
# end_frame = start_frame + 10
start_frame = 3100
end_frame = start_frame + 500


video_file = os.path.join(video_orig_path, vid_file)

vid = cv2.VideoCapture(video_file)
vid.set(1, start_frame)

j = start_frame

while j <= end_frame:
    is_read, frame = vid.read()
    if not (is_read):
        break
    frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
    # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
    for grid in grid_k:
        cv2.rectangle(frame2, (grid[0], grid[2]), (grid[1], grid[3]), (0, 0, 0), 1)
    for grid in grid_g:
        cv2.rectangle(frame2, (grid[0], grid[2]), (grid[1], grid[3]), (255, 0, 0), 1)
    cv2.rectangle(frame2, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
    cv2.rectangle(frame2, (xmin2, ymin2), (xmax2, ymax2), (255, 0, 0), 3)
    cv2.imshow('image', frame2)
    print(j)
    if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    j += 1

cv2.destroyAllWindows()


# -----------------------------------------------------------------------------------------------
# CHAMPION:  measure HSV in grids
# -----------------------------------------------------------------------------------------------

# time_slots = pd.read_csv(os.path.join(base_path, '03_datamart\\03_cycle_info.csv'))

time_slots = pd.read_csv(os.path.join(base_path, '03_datamart\\00_file_info.csv'))
time_slots.columns = ['fname','videname','cycst_orig','cyced_orig','cycst','cyced']
time_slots = time_slots.assign(dur = time_slots.cyced - time_slots.cycst)
time_slots = time_slots.assign(cycid = [cyc for cyc in range(len(time_slots))])
time_slots = time_slots[['cycid','cycst_orig','cyced_orig','dur','cycst','cyced','fname']]


rate = 1

fps = 30

hsv_k_all = []
hsv_g_all = []

cnt = 0
# for cyc in time_slots.cycid.unique():
for cyc in [4,5]:

    print(f'cycle: {cyc}')

    fname = time_slots[time_slots.cycid == cyc].fname.values[0]

    start_orig = time_slots[time_slots.cycid == cyc].cycst_orig.values[0]
    end_orig = time_slots[time_slots.cycid == cyc].cyced_orig.values[0]
    start = time_slots[time_slots.cycid == cyc].cycst.values[0]
    end = time_slots[time_slots.cycid == cyc].cyced.values[0]

    file, ext = os.path.splitext(os.path.basename(fname))
    video_file = os.path.join(video_orig_path, str(file[:-7] + '.mp4'))
    video = cv2.VideoCapture(video_file)
    frame_count = int(video.get(7))
    start_count = int(start_orig * fps)
    end_count = int(end_orig * fps)

    j = start_count
    video.set(1, j)

    i = 0

    st = time.time()
    while j <= end_count:

        # print(f'cycle: {cyc} - {j}')

        is_read, frame = video.read()
        if not (is_read):
            break
        sec = start + i / fps
        sec_orig = start_orig + i / fps
        frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
        frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

        # clone = frame2.copy()
        # cv2.rectangle(clone, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
        # cv2.rectangle(clone, (xmin2, ymin2), (xmax2, ymax2), (0, 0, 0), 3)
        # cv2.imshow("image", clone)

        hsv_k = []
        hsv_g = []

        for k, grids in enumerate(grid_k):
            h, s, v = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2HSV))
            hsv_k.append([h.mean(),s.mean(),v.mean()])

        for k, grids in enumerate(grid_g):
            h, s, v = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2HSV))
            hsv_g.append([h.mean(),s.mean(),v.mean()])

        # print(f'Keyboard {round(h.mean())} - {round(s.mean())} - {round(v.mean())} : GammaBase {round(h2.mean())} - {round(s2.mean())} - {round(v2.mean())}')

        hsv_k_all.append([cyc, sec, sec_orig] + list(itertools.chain.from_iterable(hsv_k)))
        hsv_g_all.append([cyc, sec, sec_orig] + list(itertools.chain.from_iterable(hsv_g)))

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        i += 1
        j += 1

    ed = time.time()
    print(f'cycle: {cyc} - {end_count - start_count} frames -- takes {ed - st: .2f} secs')
    cnt += 1


# ----------
hsv_k_df = pd.DataFrame(hsv_k_all)
hsv_g_df = pd.DataFrame(hsv_g_all)

coln_k = ['cycid','sec','sec_orig'] + [f'{var}_{grid}' for grid in range(len(grid_k)) for var in ['kh','ks','kv']]
hsv_k_df.columns = coln_k
hsv_k_df.reset_index(drop=True,inplace=True)

coln_g = ['cycid','sec','sec_orig'] + [f'{var}_{grid}' for grid in range(len(grid_g)) for var in ['gh','gs','gv']]
hsv_g_df.columns = coln_g
hsv_g_df.reset_index(drop=True,inplace=True)

hsv_k_df.to_csv(os.path.join(base_path, '03_datamart\\hsv_k_df_champ.csv'), index=False)
hsv_g_df.to_csv(os.path.join(base_path, '03_datamart\\hsv_g_df_champ.csv'), index=False)



#################################################################################################
# -----------------------------------------------------------------------------------------------
# NON-CHAMPION:  set grid
# -----------------------------------------------------------------------------------------------

width = 36

# objective area:  base
# Keyboard
xmin = 280-130
ymin = 45+30
xmax = xmin + width
ymax = ymin + width

# # GammaBase
xmin2 = 270-120
ymin2 = 145+50
xmax2 = xmin2 + width
ymax2 = ymin2 + width


# # -----------
# # generate grid to cover Keyboard and GammaBase and other area
# x = list(range(0, 640, 30))[1:-1][::2]
# y = [5] + list(range(-5, 480, 30))[2:-1][::2]
#
# grid_k0 = list(itertools.product(x,y))
#
# grids = []
# for i in range(len(grid_k0)):
#     grids.append((grid_k0[i][0], grid_k0[i][0]+30, grid_k0[i][1], grid_k0[i][1]+30))
#
# # add GammaBase
# grids.append((xmin, xmax, ymin, ymax))


# ----------
step = int(width / 2)
# add slightly adjacent area: keyboard
x2 = list(range(xmin-step*2, xmax+step*1, step))
y2 = list(range(ymin-step*2, ymax+step*1, step))

grid_k0 = list(itertools.product(x2,y2))

grid_k = []
for i in range(len(grid_k0)):
    grid_k.append((grid_k0[i][0], grid_k0[i][0]+width, grid_k0[i][1], grid_k0[i][1]+width))

# add slightly adjacent area: GammaBase
x3 = list(range(xmin2-step*2, xmax2+step*1, step))
y3 = list(range(ymin2-step*2, ymax2+step*1, step))

grid_g0 = list(itertools.product(x3,y3))

grid_g = []
for i in range(len(grid_g0)):
    grid_g.append((grid_g0[i][0], grid_g0[i][0]+width, grid_g0[i][1], grid_g0[i][1]+width))


# ----------
print(len(grid_k))
print(grid_k)

print(len(grid_g))
print(grid_g)


# -----------------------------------------------------------------------------------------------
# NON-CHAMPION:  check the grid
# -----------------------------------------------------------------------------------------------

vid_file = 'Gamma_Assembly_192_168_32_69_1_20201130110000_10min.mp4'
start_frame = 3400
end_frame = start_frame + 700

# vid_file = 'Gamma_Assembly_192_168_32_69_1_20201130180000_10min.mp4'
# start_frame = 1400
# end_frame = start_frame + 700


rate = 1

video_file = os.path.join(video_orig_path, vid_file)

vid = cv2.VideoCapture(video_file)
vid.set(1, start_frame)

j = start_frame

while j <= end_frame:
    is_read, frame = vid.read()
    if not (is_read):
        break
    frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
    # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
    for grid in grid_k:
        cv2.rectangle(frame2, (grid[0], grid[2]), (grid[1], grid[3]), (0, 0, 0), 1)
    for grid in grid_g:
        cv2.rectangle(frame2, (grid[0], grid[2]), (grid[1], grid[3]), (255, 0, 0), 1)
    cv2.rectangle(frame2, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
    cv2.rectangle(frame2, (xmin2, ymin2), (xmax2, ymax2), (255, 0, 0), 3)
    cv2.imshow('image', frame2)
    print(j)
    if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    j += 1

cv2.destroyAllWindows()


# -----------------------------------------------------------------------------------------------
# NON-CHAMPION:  measure HSV in grids
# -----------------------------------------------------------------------------------------------

# time_slots = pd.read_csv(os.path.join(base_path, '03_datamart\\03_cycle_info.csv'))

time_slots = pd.read_csv(os.path.join(base_path, '03_datamart\\00_file_info.csv'))
time_slots.columns = ['fname','videname','cycst_orig','cyced_orig','cycst','cyced']
time_slots = time_slots.assign(dur = time_slots.cyced - time_slots.cycst)
time_slots = time_slots.assign(cycid = [cyc for cyc in range(len(time_slots))])
time_slots = time_slots[['cycid','cycst_orig','cyced_orig','dur','cycst','cyced','fname']]


rate = 1

fps = 30

hsv_k_all = []
hsv_g_all = []

# for cyc in time_slots.cycid.unique():
for cyc in [0,1,2,3]:

    print(f'cycle: {cyc}')

    fname = time_slots[time_slots.cycid == cyc].fname.values[0]

    start_orig = time_slots[time_slots.cycid == cyc].cycst_orig.values[0]
    end_orig = time_slots[time_slots.cycid == cyc].cyced_orig.values[0]
    start = time_slots[time_slots.cycid == cyc].cycst.values[0]
    end = time_slots[time_slots.cycid == cyc].cyced.values[0]

    file, ext = os.path.splitext(os.path.basename(fname))
    video_file = os.path.join(video_orig_path, str(file[:-7] + '.mp4'))
    video = cv2.VideoCapture(video_file)
    frame_count = int(video.get(7))
    start_count = int(start_orig * fps)
    end_count = int(end_orig * fps)

    j = start_count
    video.set(1, j)

    i = 0

    st = time.time()
    while j <= end_count:

        is_read, frame = video.read()
        if not (is_read):
            break
        sec = start + i / fps
        sec_orig = start_orig + i / fps
        frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
        # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

        # print(f'cycle: {cyc} - {j} - {sec}')

        # clone = frame2.copy()
        # cv2.rectangle(clone, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
        # cv2.rectangle(clone, (xmin2, ymin2), (xmax2, ymax2), (0, 0, 0), 3)
        # cv2.imshow("image", clone)

        hsv_k = []
        hsv_g = []

        for k, grids in enumerate(grid_k):
            h, s, v = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2HSV))
            hsv_k.append([h.mean(),s.mean(),v.mean()])

        for k, grids in enumerate(grid_g):
            h, s, v = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2HSV))
            hsv_g.append([h.mean(),s.mean(),v.mean()])

        # print(f'Keyboard {round(h.mean())} - {round(s.mean())} - {round(v.mean())} : GammaBase {round(h2.mean())} - {round(s2.mean())} - {round(v2.mean())}')

        hsv_k_all.append([cyc, sec, sec_orig] + list(itertools.chain.from_iterable(hsv_k)))
        hsv_g_all.append([cyc, sec, sec_orig] + list(itertools.chain.from_iterable(hsv_g)))

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        i += 1
        j += 1

    ed = time.time()
    print(f'cycle: {cyc} - {end_count - start_count} frames -- takes {ed - st: .2f} secs')


# ----------
hsv_k_df = pd.DataFrame(hsv_k_all)
hsv_g_df = pd.DataFrame(hsv_g_all)

coln_k = ['cycid','sec','sec_orig'] + [f'{var}_{grid}' for grid in range(len(grid_k)) for var in ['kh','ks','kv']]
hsv_k_df.columns = coln_k
hsv_k_df.reset_index(drop=True,inplace=True)

coln_g = ['cycid','sec','sec_orig'] + [f'{var}_{grid}' for grid in range(len(grid_g)) for var in ['gh','gs','gv']]
hsv_g_df.columns = coln_g
hsv_g_df.reset_index(drop=True,inplace=True)

hsv_k_df.to_csv(os.path.join(base_path, '03_datamart\\hsv_k_df_nonchamp.csv'), index=False)
hsv_g_df.to_csv(os.path.join(base_path, '03_datamart\\hsv_g_df_nonchamp.csv'), index=False)


#################################################################################################
# -----------------------------------------------------------------------------------------------
# 0420 PM video:  set grid
# -----------------------------------------------------------------------------------------------

width = 22

# objective area:  base
# Keyboard
xmin = 340
ymin = 15
xmax = xmin + width
ymax = ymin + width

# xmin = 220
# ymin = 15
# xmax = xmin + width
# ymax = ymin + width

# xmin = 351
# ymin = 55
# xmax = xmin + 11
# ymax = ymin + 11

# # GammaBase
xmin2 = 340
ymin2 = 140
xmax2 = xmin2 + width
ymax2 = ymin2 + width


# # -----------
# # generate grid to cover Keyboard and GammaBase and other area
# x = list(range(0, 640, 30))[1:-1][::2]
# y = [5] + list(range(-5, 480, 30))[2:-1][::2]
#
# grid_k0 = list(itertools.product(x,y))
#
# grids = []
# for i in range(len(grid_k0)):
#     grids.append((grid_k0[i][0], grid_k0[i][0]+30, grid_k0[i][1], grid_k0[i][1]+30))
#
# # add GammaBase
# grids.append((xmin, xmax, ymin, ymax))


# ----------
step = int(width / 2)
# add slightly adjacent area: keyboard
x2 = list(range(xmin-step*2, xmax+step*1, step))
# y2 = list(range(ymin-step*2, ymax+step*1, step))
y2 = list(range(ymin-step*1, ymax+step*1, step))

grid_k0 = list(itertools.product(x2,y2))

grid_k = []
for i in range(len(grid_k0)):
    grid_k.append((grid_k0[i][0], grid_k0[i][0]+width, grid_k0[i][1], grid_k0[i][1]+width))

# add slightly adjacent area: GammaBase
x3 = list(range(xmin2-step*2, xmax2+step*1, step))
y3 = list(range(ymin2-step*2, ymax2+step*1, step))

grid_g0 = list(itertools.product(x3,y3))

grid_g = []
for i in range(len(grid_g0)):
    grid_g.append((grid_g0[i][0], grid_g0[i][0]+width, grid_g0[i][1], grid_g0[i][1]+width))


# ----------
print(len(grid_k))
print(grid_k)

print(len(grid_g))
print(grid_g)


# -----------------------------------------------------------------------------------------------
# 0420 PM video:  check the grid
# -----------------------------------------------------------------------------------------------

# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210303165000_10min.mp4'
# start_frame = 2000
# end_frame = start_frame + 300

# start_frame = 5000
# end_frame = start_frame + 10

# something on work station
# start_frame = 2800
# end_frame = start_frame + 200


# vid_file = 'Gamma_Assembly_192_168_32_69_1_20210223134400_10min.mp4'
# start_frame = 1
# end_frame = start_frame + 10
# start_frame = 3100
# end_frame = start_frame + 500

rate = 1

# vid_file = 'Gamma_Assembly_192_168_32_69_1_20201130180000_10min.mp4'

vid_file = 'test_192_168_32_70_20210420133211_60min.mp4'
# start_frame = 1
# end_frame = start_frame + 10
start_frame = 9000
end_frame = start_frame + 500


video_file = os.path.join(video_orig_path, vid_file)

vid = cv2.VideoCapture(video_file)
vid.set(1, start_frame)

j = start_frame

while j <= end_frame:
    is_read, frame = vid.read()
    if not (is_read):
        break
    frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
    # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
    for grid in grid_k:
        cv2.rectangle(frame2, (grid[0], grid[2]), (grid[1], grid[3]), (0, 0, 0), 1)
    for grid in grid_g:
        cv2.rectangle(frame2, (grid[0], grid[2]), (grid[1], grid[3]), (255, 0, 0), 1)
    cv2.rectangle(frame2, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
    cv2.rectangle(frame2, (xmin2, ymin2), (xmax2, ymax2), (255, 0, 0), 3)
    cv2.imshow('image', frame2)
    print(j)
    if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    j += 1

cv2.destroyAllWindows()


# -----------------------------------------------------------------------------------------------
# 0420 PM video:  measure HSV in grids
# -----------------------------------------------------------------------------------------------

time_slots = pd.read_csv(os.path.join(base_path, '03_datamart\\00_file_info.csv'))
time_slots.columns = ['fname','videname','cycst_orig','cyced_orig','cycst','cyced']
time_slots = time_slots.assign(dur = time_slots.cyced - time_slots.cycst)
time_slots = time_slots.assign(cycid = [cyc for cyc in range(len(time_slots))])
time_slots = time_slots[['cycid','cycst_orig','cyced_orig','dur','cycst','cyced','fname']]


rate = 1

fps = 30

hsv_k_all = []
hsv_g_all = []

cnt = 0
for cyc in time_slots.cycid.unique():
# for cyc in [4,5]:

    print(f'cycle: {cyc}')

    fname = time_slots[time_slots.cycid == cyc].fname.values[0]

    start_orig = time_slots[time_slots.cycid == cyc].cycst_orig.values[0]
    end_orig = time_slots[time_slots.cycid == cyc].cyced_orig.values[0]
    start = time_slots[time_slots.cycid == cyc].cycst.values[0]
    end = time_slots[time_slots.cycid == cyc].cyced.values[0]

    file, ext = os.path.splitext(os.path.basename(fname))
    video_file = os.path.join(video_orig_path, str(file[:-7] + '.mp4'))
    video = cv2.VideoCapture(video_file)
    frame_count = int(video.get(7))
    start_count = int(start_orig * fps)
    end_count = int(end_orig * fps)

    j = start_count
    video.set(1, j)

    i = 0

    st = time.time()
    while j <= end_count:

        # print(f'cycle: {cyc} - {j}')

        is_read, frame = video.read()
        if not (is_read):
            break
        sec = start + i / fps
        sec_orig = start_orig + i / fps
        frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
        # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

        clone = frame2.copy()
        cv2.rectangle(clone, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
        cv2.rectangle(clone, (xmin2, ymin2), (xmax2, ymax2), (0, 0, 0), 3)
        cv2.imshow("image", clone)

        hsv_k = []
        hsv_g = []

        for k, grids in enumerate(grid_k):
            h, s, v = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2HSV))
            # h, s, v = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2LAB))
            hsv_k.append([h.mean(),s.mean(),v.mean()])

        for k, grids in enumerate(grid_g):
            h, s, v = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2HSV))
            # h, s, v = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2LAB))
            hsv_g.append([h.mean(),s.mean(),v.mean()])

        # print(f'Keyboard {round(h.mean())} - {round(s.mean())} - {round(v.mean())} : GammaBase {round(h2.mean())} - {round(s2.mean())} - {round(v2.mean())}')

        hsv_k_all.append([cyc, sec, sec_orig] + list(itertools.chain.from_iterable(hsv_k)))
        hsv_g_all.append([cyc, sec, sec_orig] + list(itertools.chain.from_iterable(hsv_g)))

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        i += 1
        j += 1

    ed = time.time()
    print(f'cycle: {cyc} - {end_count - start_count} frames -- takes {ed - st: .2f} secs')
    cnt += 1


# ----------
hsv_k_df = pd.DataFrame(hsv_k_all)
hsv_g_df = pd.DataFrame(hsv_g_all)

coln_k = ['cycid','sec','sec_orig'] + [f'{var}_{grid}' for grid in range(len(grid_k)) for var in ['kh','ks','kv']]
hsv_k_df.columns = coln_k
hsv_k_df.reset_index(drop=True,inplace=True)

coln_g = ['cycid','sec','sec_orig'] + [f'{var}_{grid}' for grid in range(len(grid_g)) for var in ['gh','gs','gv']]
hsv_g_df.columns = coln_g
hsv_g_df.reset_index(drop=True,inplace=True)

# hsv_k_df.to_csv(os.path.join(base_path, '03_datamart\\hsv_k_df_0420pm.csv'), index=False)
# hsv_g_df.to_csv(os.path.join(base_path, '03_datamart\\hsv_g_df_0420pm.csv'), index=False)
# hsv_k_df.to_csv(os.path.join(base_path, '03_datamart\\lab_k_df_0420pm.csv'), index=False)
# hsv_g_df.to_csv(os.path.join(base_path, '03_datamart\\lab_g_df_0420pm.csv'), index=False)


#################################################################################################
# -----------------------------------------------------------------------------------------------
# HSV analysis
# -----------------------------------------------------------------------------------------------

# hsv_k = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_k_df_champ.csv'))
# hsv_g = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_g_df_champ.csv'))

# hsv_k = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_k_df_nonchamp.csv'))
# hsv_g = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_g_df_nonchamp.csv'))

# hsv_k = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_k_df_0420pm.csv'))
# hsv_g = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_g_df_0420pm.csv'))

hsv_k = pd.read_csv(os.path.join(base_path, '03_datamart\\lab_k_df_0420pm.csv'))
hsv_g = pd.read_csv(os.path.join(base_path, '03_datamart\\lab_g_df_0420pm.csv'))

print(hsv_k.sec_orig.describe())
print(hsv_k.sec.describe())

print(hsv_g.sec_orig.describe())
print(hsv_g.sec.describe())

hsv_k_df_c2 = hsv_k.copy()
hsv_g_df_c2 = hsv_g.copy()


# -----------
# cycinfo_gt = pd.read_csv(os.path.join(base_path, '03_datamart\\03_time_slots_around_cycle_start_gt.csv'))


# ----------
# cycinfo_gt = pd.read_csv(os.path.join(base_path, '01_data\\subcycle_info\\03_cycle_info_02_screw_gt_20210521.txt'), sep='\t')
# hsv_k_df_c2 = utils.filter_within_cycle(cycinfo=cycinfo_gt, dfobj=hsv_k, adjust_before=0, adjust_after=0)
# hsv_g_df_c2 = utils.filter_within_cycle(cycinfo=cycinfo_gt, dfobj=hsv_g, adjust_before=0, adjust_after=0)


# ----------
# cycinfo_gt = pd.read_csv(os.path.join(base_path, '01_data\\subcycle_info\\03_cycle_info_02_screw_gt_20210521.txt'), sep='\t')
# time_slots_before = utils.time_slots_before_start(cycinfo=cycinfo_gt, file_info=file_info, take_secs=5)
# time_slots_after = utils.time_slots_after_start(cycinfo=cycinfo_gt, file_info=file_info, take_secs=5)

# cycinfo_gt = pd.read_csv(os.path.join(base_path, '01_data\\subcycle_info\\subcycle_info_PCDGamma_pcd0420_pm_k.txt'), sep='\t')
cycinfo_gt = pd.read_csv(os.path.join(base_path, '01_data\\subcycle_info\\subcycle_info_PCDGamma_pcd0420_pm_g.txt'), sep='\t')
time_slots_before = utils.time_slots_before_start(cycinfo=cycinfo_gt, file_info=file_info, take_secs=5)
time_slots_after = utils.time_slots_after_start(cycinfo=cycinfo_gt, file_info=file_info, take_secs=5)


hsv_k_df_c2 = utils.filter_within_cycle(cycinfo=time_slots_before, dfobj=hsv_k, adjust_before=0, adjust_after=0)
hsv_g_df_c2 = utils.filter_within_cycle(cycinfo=time_slots_before, dfobj=hsv_g, adjust_before=0, adjust_after=0)

# hsv_k_df_c2 = utils.filter_within_cycle(cycinfo=time_slots_after, dfobj=hsv_k, adjust_before=0, adjust_after=0)
# hsv_g_df_c2 = utils.filter_within_cycle(cycinfo=time_slots_after, dfobj=hsv_g, adjust_before=0, adjust_after=0)


# ----------
# objective grid
# print(grid_k)
# print(grid_g)
#
# print(grid_k[12])
# print(grid_g[12])
#
# hsv_k_df_c[['kh_12']].describe()
# hsv_k_df_c[['ks_12']].describe()
# hsv_k_df_c[['kv_12']].describe()


fig, axes = plt.subplots(5, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_k_df_c2[['kv_0']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][0].hist(hsv_k_df_c2[['kv_1']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][0].hist(hsv_k_df_c2[['kv_2']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][0].hist(hsv_k_df_c2[['kv_3']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][0].hist(hsv_k_df_c2[['kv_4']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][1].hist(hsv_k_df_c2[['kv_5']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][1].hist(hsv_k_df_c2[['kv_6']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][1].hist(hsv_k_df_c2[['kv_7']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][1].hist(hsv_k_df_c2[['kv_8']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][1].hist(hsv_k_df_c2[['kv_9']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][2].hist(hsv_k_df_c2[['kv_10']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][2].hist(hsv_k_df_c2[['kv_11']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][2].hist(hsv_k_df_c2[['kv_12']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][2].hist(hsv_k_df_c2[['kv_13']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][2].hist(hsv_k_df_c2[['kv_14']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][3].hist(hsv_k_df_c2[['kv_15']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][3].hist(hsv_k_df_c2[['kv_16']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][3].hist(hsv_k_df_c2[['kv_17']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][3].hist(hsv_k_df_c2[['kv_18']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][3].hist(hsv_k_df_c2[['kv_19']].values, range(0, 255, 5), rwidth=0.8, color='red')
# axes[0][4].hist(hsv_k_df_c2[['kv_20']].values, range(0, 255, 5), rwidth=0.8, color='red')
# axes[1][4].hist(hsv_k_df_c2[['kv_21']].values, range(0, 255, 5), rwidth=0.8, color='red')
# axes[2][4].hist(hsv_k_df_c2[['kv_22']].values, range(0, 255, 5), rwidth=0.8, color='red')
# axes[3][4].hist(hsv_k_df_c2[['kv_23']].values, range(0, 255, 5), rwidth=0.8, color='red')
# axes[4][4].hist(hsv_k_df_c2[['kv_24']].values, range(0, 255, 5), rwidth=0.8, color='red')
fig.suptitle(f'Keyboard Area Grid: Histogram of V value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_Keyboard_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_around_cycle_start_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_before_cycle_start_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_after_cycle_start_V.png'))
plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_Keyboard_before_cycle_start_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_Keyboard_after_cycle_start_V.png'))
plt.show()
plt.close()

fig, axes = plt.subplots(5, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_k_df_c2[['kh_0']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][0].hist(hsv_k_df_c2[['kh_1']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][0].hist(hsv_k_df_c2[['kh_2']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][0].hist(hsv_k_df_c2[['kh_3']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][0].hist(hsv_k_df_c2[['kh_4']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][1].hist(hsv_k_df_c2[['kh_5']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][1].hist(hsv_k_df_c2[['kh_6']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][1].hist(hsv_k_df_c2[['kh_7']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][1].hist(hsv_k_df_c2[['kh_8']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][1].hist(hsv_k_df_c2[['kh_9']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][2].hist(hsv_k_df_c2[['kh_10']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][2].hist(hsv_k_df_c2[['kh_11']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][2].hist(hsv_k_df_c2[['kh_12']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][2].hist(hsv_k_df_c2[['kh_13']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][2].hist(hsv_k_df_c2[['kh_14']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][3].hist(hsv_k_df_c2[['kh_15']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][3].hist(hsv_k_df_c2[['kh_16']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][3].hist(hsv_k_df_c2[['kh_17']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][3].hist(hsv_k_df_c2[['kh_18']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][3].hist(hsv_k_df_c2[['kh_19']].values, range(0, 180, 5), rwidth=0.8, color='blue')
# axes[0][4].hist(hsv_k_df_c2[['kh_20']].values, range(0, 180, 5), rwidth=0.8, color='blue')
# axes[1][4].hist(hsv_k_df_c2[['kh_21']].values, range(0, 180, 5), rwidth=0.8, color='blue')
# axes[2][4].hist(hsv_k_df_c2[['kh_22']].values, range(0, 180, 5), rwidth=0.8, color='blue')
# axes[3][4].hist(hsv_k_df_c2[['kh_23']].values, range(0, 180, 5), rwidth=0.8, color='blue')
# axes[4][4].hist(hsv_k_df_c2[['kh_24']].values, range(0, 180, 5), rwidth=0.8, color='blue')
fig.suptitle(f'Keyboard Area Grid: Histogram of H value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_Keyboard_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_around_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_before_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_after_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_Keyboard_before_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_Keyboard_after_cycle_start_H.png'))
plt.show()
plt.close()


# fig, axes = plt.subplots(5, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
fig, axes = plt.subplots(4, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_k_df_c2[['ks_0']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][0].hist(hsv_k_df_c2[['ks_1']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][0].hist(hsv_k_df_c2[['ks_2']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][0].hist(hsv_k_df_c2[['ks_3']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][0].hist(hsv_k_df_c2[['ks_4']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][1].hist(hsv_k_df_c2[['ks_5']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][1].hist(hsv_k_df_c2[['ks_6']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][1].hist(hsv_k_df_c2[['ks_7']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][1].hist(hsv_k_df_c2[['ks_8']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][1].hist(hsv_k_df_c2[['ks_9']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][2].hist(hsv_k_df_c2[['ks_10']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][2].hist(hsv_k_df_c2[['ks_11']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][2].hist(hsv_k_df_c2[['ks_12']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][2].hist(hsv_k_df_c2[['ks_13']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][2].hist(hsv_k_df_c2[['ks_14']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][3].hist(hsv_k_df_c2[['ks_15']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][3].hist(hsv_k_df_c2[['ks_16']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][3].hist(hsv_k_df_c2[['ks_17']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][3].hist(hsv_k_df_c2[['ks_18']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][3].hist(hsv_k_df_c2[['ks_19']].values, range(0, 255, 5), rwidth=0.8, color='orange')
# axes[0][4].hist(hsv_k_df_c2[['ks_20']].values, range(0, 255, 5), rwidth=0.8, color='orange')
# axes[1][4].hist(hsv_k_df_c2[['ks_21']].values, range(0, 255, 5), rwidth=0.8, color='orange')
# axes[2][4].hist(hsv_k_df_c2[['ks_22']].values, range(0, 255, 5), rwidth=0.8, color='orange')
# axes[3][4].hist(hsv_k_df_c2[['ks_23']].values, range(0, 255, 5), rwidth=0.8, color='orange')
# axes[4][4].hist(hsv_k_df_c2[['ks_24']].values, range(0, 255, 5), rwidth=0.8, color='orange')
fig.suptitle(f'Keyboard Area Grid: Histogram of S value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_Keyboard_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_around_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_before_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_Keyboard_after_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_Keyboard_before_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NOnChampion_Keyboard_after_cycle_start_S.png'))
plt.show()
plt.close()


fig, axes = plt.subplots(5, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_g_df_c2[['gv_0']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][0].hist(hsv_g_df_c2[['gv_1']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][0].hist(hsv_g_df_c2[['gv_2']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][0].hist(hsv_g_df_c2[['gv_3']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][0].hist(hsv_g_df_c2[['gv_4']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][1].hist(hsv_g_df_c2[['gv_5']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][1].hist(hsv_g_df_c2[['gv_6']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][1].hist(hsv_g_df_c2[['gv_7']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][1].hist(hsv_g_df_c2[['gv_8']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][1].hist(hsv_g_df_c2[['gv_9']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][2].hist(hsv_g_df_c2[['gv_10']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][2].hist(hsv_g_df_c2[['gv_11']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][2].hist(hsv_g_df_c2[['gv_12']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][2].hist(hsv_g_df_c2[['gv_13']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][2].hist(hsv_g_df_c2[['gv_14']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][3].hist(hsv_g_df_c2[['gv_15']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][3].hist(hsv_g_df_c2[['gv_16']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][3].hist(hsv_g_df_c2[['gv_17']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][3].hist(hsv_g_df_c2[['gv_18']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][3].hist(hsv_g_df_c2[['gv_19']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][4].hist(hsv_g_df_c2[['gv_20']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][4].hist(hsv_g_df_c2[['gv_21']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][4].hist(hsv_g_df_c2[['gv_22']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][4].hist(hsv_g_df_c2[['gv_23']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][4].hist(hsv_g_df_c2[['gv_24']].values, range(0, 255, 5), rwidth=0.8, color='red')
fig.suptitle(f'GammaBase Area Grid: Histogram of V value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_GammaBase_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_around_cycle_start_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_before_cycle_start_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_after_cycle_start_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_GammaBase_before_cycle_start_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_GammaBase_after_cycle_start_V.png'))
plt.show()
plt.close()


fig, axes = plt.subplots(5, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_g_df_c2[['gh_0']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][0].hist(hsv_g_df_c2[['gh_1']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][0].hist(hsv_g_df_c2[['gh_2']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][0].hist(hsv_g_df_c2[['gh_3']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][0].hist(hsv_g_df_c2[['gh_4']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][1].hist(hsv_g_df_c2[['gh_5']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][1].hist(hsv_g_df_c2[['gh_6']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][1].hist(hsv_g_df_c2[['gh_7']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][1].hist(hsv_g_df_c2[['gh_8']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][1].hist(hsv_g_df_c2[['gh_9']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][2].hist(hsv_g_df_c2[['gh_10']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][2].hist(hsv_g_df_c2[['gh_11']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][2].hist(hsv_g_df_c2[['gh_12']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][2].hist(hsv_g_df_c2[['gh_13']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][2].hist(hsv_g_df_c2[['gh_14']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][3].hist(hsv_g_df_c2[['gh_15']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][3].hist(hsv_g_df_c2[['gh_16']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][3].hist(hsv_g_df_c2[['gh_17']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][3].hist(hsv_g_df_c2[['gh_18']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][3].hist(hsv_g_df_c2[['gh_19']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][4].hist(hsv_g_df_c2[['gh_20']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][4].hist(hsv_g_df_c2[['gh_21']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][4].hist(hsv_g_df_c2[['gh_22']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][4].hist(hsv_g_df_c2[['gh_23']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][4].hist(hsv_g_df_c2[['gh_24']].values, range(0, 180, 5), rwidth=0.8, color='blue')
fig.suptitle(f'GammaBase Area Grid: Histogram of H value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_GammaBase_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_around_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_before_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_after_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_GammaBase_before_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_GammaBase_after_cycle_start_H.png'))
plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_before_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_after_cycle_start_H.png'))
plt.show()
plt.close()


fig, axes = plt.subplots(5, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_g_df_c2[['gs_0']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][0].hist(hsv_g_df_c2[['gs_1']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][0].hist(hsv_g_df_c2[['gs_2']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][0].hist(hsv_g_df_c2[['gs_3']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][0].hist(hsv_g_df_c2[['gs_4']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][1].hist(hsv_g_df_c2[['gs_5']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][1].hist(hsv_g_df_c2[['gs_6']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][1].hist(hsv_g_df_c2[['gs_7']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][1].hist(hsv_g_df_c2[['gs_8']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][1].hist(hsv_g_df_c2[['gs_9']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][2].hist(hsv_g_df_c2[['gs_10']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][2].hist(hsv_g_df_c2[['gs_11']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][2].hist(hsv_g_df_c2[['gs_12']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][2].hist(hsv_g_df_c2[['gs_13']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][2].hist(hsv_g_df_c2[['gs_14']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][3].hist(hsv_g_df_c2[['gs_15']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][3].hist(hsv_g_df_c2[['gs_16']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][3].hist(hsv_g_df_c2[['gs_17']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][3].hist(hsv_g_df_c2[['gs_18']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][3].hist(hsv_g_df_c2[['gs_19']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][4].hist(hsv_g_df_c2[['gs_20']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][4].hist(hsv_g_df_c2[['gs_21']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][4].hist(hsv_g_df_c2[['gs_22']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][4].hist(hsv_g_df_c2[['gs_23']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][4].hist(hsv_g_df_c2[['gs_24']].values, range(0, 255, 5), rwidth=0.8, color='orange')
fig.suptitle(f'Keyboard Area Grid: Histogram of S value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_GammaBase_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_around_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_before_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\Champion_GammaBase_after_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_GammaBase_before_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\NonChampion_GammaBase_after_cycle_start_S.png'))
plt.show()
plt.close()



#################################################################################################
# -----------------------------------------------------------------------------------------------
# HSV analysis: 0420 pm
# -----------------------------------------------------------------------------------------------

# hsv_k = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_k_df_champ.csv'))
# hsv_g = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_g_df_champ.csv'))

# hsv_k = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_k_df_nonchamp.csv'))
# hsv_g = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_g_df_nonchamp.csv'))

hsv_k = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_k_df_0420pm.csv'))
hsv_g = pd.read_csv(os.path.join(base_path, '03_datamart\\hsv_g_df_0420pm.csv'))

# hsv_k = pd.read_csv(os.path.join(base_path, '03_datamart\\lab_k_df_0420pm.csv'))
# hsv_g = pd.read_csv(os.path.join(base_path, '03_datamart\\lab_g_df_0420pm.csv'))


print(hsv_k.sec_orig.describe())
print(hsv_k.sec.describe())

print(hsv_g.sec_orig.describe())
print(hsv_g.sec.describe())

hsv_k_df_c2 = hsv_k.copy()
hsv_g_df_c2 = hsv_g.copy()




# -----------
# cycinfo_gt = pd.read_csv(os.path.join(base_path, '03_datamart\\03_time_slots_around_cycle_start_gt.csv'))


# ----------
# cycinfo_gt = pd.read_csv(os.path.join(base_path, '01_data\\subcycle_info\\subcycle_info_PCDGamma_pcd0420_pm.txt'), sep='\t')
# hsv_k_df_c2 = utils.filter_within_cycle(cycinfo=cycinfo_gt, dfobj=hsv_k, adjust_before=0, adjust_after=0)
# hsv_g_df_c2 = utils.filter_within_cycle(cycinfo=cycinfo_gt, dfobj=hsv_g, adjust_before=0, adjust_after=0)


# ----------
cycinfo_gt = pd.read_csv(os.path.join(base_path, '01_data\\subcycle_info\\subcycle_info_PCDGamma_pcd0420_pm_k.txt'), sep='\t')
# cycinfo_gt = pd.read_csv(os.path.join(base_path, '01_data\\subcycle_info\\subcycle_info_PCDGamma_pcd0420_pm_g.txt'), sep='\t')

time_slots_before = utils.time_slots_before_start(cycinfo=cycinfo_gt, file_info=file_info, take_secs=5)
time_slots_after = utils.time_slots_after_start(cycinfo=cycinfo_gt, file_info=file_info, take_secs=5)


# hsv_k_df_c2 = utils.filter_within_cycle(cycinfo=time_slots_before, dfobj=hsv_k, adjust_before=0, adjust_after=0)
# hsv_g_df_c2 = utils.filter_within_cycle(cycinfo=time_slots_before, dfobj=hsv_g, adjust_before=0, adjust_after=0)

hsv_k_df_c2 = utils.filter_within_cycle(cycinfo=time_slots_after, dfobj=hsv_k, adjust_before=0, adjust_after=0)
hsv_g_df_c2 = utils.filter_within_cycle(cycinfo=time_slots_after, dfobj=hsv_g, adjust_before=0, adjust_after=0)


# ----------
# objective grid
# print(grid_k)
# print(grid_g)
#
# print(grid_k[12])
# print(grid_g[12])
#
# hsv_k_df_c[['kh_12']].describe()
# hsv_k_df_c[['ks_12']].describe()
# hsv_k_df_c[['kv_12']].describe()


fig, axes = plt.subplots(4, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_k_df_c2[['kv_0']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][0].hist(hsv_k_df_c2[['kv_1']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][0].hist(hsv_k_df_c2[['kv_2']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][0].hist(hsv_k_df_c2[['kv_3']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][1].hist(hsv_k_df_c2[['kv_4']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][1].hist(hsv_k_df_c2[['kv_5']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][1].hist(hsv_k_df_c2[['kv_6']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][1].hist(hsv_k_df_c2[['kv_7']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][2].hist(hsv_k_df_c2[['kv_8']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][2].hist(hsv_k_df_c2[['kv_9']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][2].hist(hsv_k_df_c2[['kv_10']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][2].hist(hsv_k_df_c2[['kv_11']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][3].hist(hsv_k_df_c2[['kv_12']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][3].hist(hsv_k_df_c2[['kv_13']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][3].hist(hsv_k_df_c2[['kv_14']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][3].hist(hsv_k_df_c2[['kv_15']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][4].hist(hsv_k_df_c2[['kv_16']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][4].hist(hsv_k_df_c2[['kv_17']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][4].hist(hsv_k_df_c2[['kv_18']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][4].hist(hsv_k_df_c2[['kv_19']].values, range(0, 255, 5), rwidth=0.8, color='red')
fig.suptitle(f'Keyboard Area Grid: Histogram of V value.png', fontsize=15)
plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_Keyboard_before_cycle_start_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_Keyboard_after_cycle_start_V.png'))
plt.show()
plt.close()

fig, axes = plt.subplots(4, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_k_df_c2[['kh_0']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][0].hist(hsv_k_df_c2[['kh_1']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][0].hist(hsv_k_df_c2[['kh_2']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][0].hist(hsv_k_df_c2[['kh_3']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][1].hist(hsv_k_df_c2[['kh_4']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][1].hist(hsv_k_df_c2[['kh_5']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][1].hist(hsv_k_df_c2[['kh_6']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][1].hist(hsv_k_df_c2[['kh_7']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][2].hist(hsv_k_df_c2[['kh_8']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][2].hist(hsv_k_df_c2[['kh_9']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][2].hist(hsv_k_df_c2[['kh_10']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][2].hist(hsv_k_df_c2[['kh_11']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][3].hist(hsv_k_df_c2[['kh_12']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][3].hist(hsv_k_df_c2[['kh_13']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][3].hist(hsv_k_df_c2[['kh_14']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][3].hist(hsv_k_df_c2[['kh_15']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][4].hist(hsv_k_df_c2[['kh_16']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][4].hist(hsv_k_df_c2[['kh_17']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][4].hist(hsv_k_df_c2[['kh_18']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][4].hist(hsv_k_df_c2[['kh_19']].values, range(0, 180, 5), rwidth=0.8, color='blue')
fig.suptitle(f'Keyboard Area Grid: Histogram of H value.png', fontsize=15)
plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_Keyboard_before_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_Keyboard_after_cycle_start_H.png'))
plt.show()
plt.close()


fig, axes = plt.subplots(4, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_k_df_c2[['ks_0']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][0].hist(hsv_k_df_c2[['ks_1']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][0].hist(hsv_k_df_c2[['ks_2']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][0].hist(hsv_k_df_c2[['ks_3']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][1].hist(hsv_k_df_c2[['ks_4']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][1].hist(hsv_k_df_c2[['ks_5']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][1].hist(hsv_k_df_c2[['ks_6']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][1].hist(hsv_k_df_c2[['ks_7']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][2].hist(hsv_k_df_c2[['ks_8']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][2].hist(hsv_k_df_c2[['ks_9']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][2].hist(hsv_k_df_c2[['ks_10']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][2].hist(hsv_k_df_c2[['ks_11']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][3].hist(hsv_k_df_c2[['ks_12']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][3].hist(hsv_k_df_c2[['ks_13']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][3].hist(hsv_k_df_c2[['ks_14']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][3].hist(hsv_k_df_c2[['ks_15']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][4].hist(hsv_k_df_c2[['ks_16']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][4].hist(hsv_k_df_c2[['ks_17']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][4].hist(hsv_k_df_c2[['ks_18']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][4].hist(hsv_k_df_c2[['ks_19']].values, range(0, 255, 5), rwidth=0.8, color='orange')
# fig.suptitle(f'Keyboard Area Grid: Histogram of S value.png', fontsize=15)
plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_Keyboard_before_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_Keyboard_after_cycle_start_S.png'))
plt.show()
plt.close()


# --------------------------------
fig, axes = plt.subplots(5, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_g_df_c2[['gv_0']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][0].hist(hsv_g_df_c2[['gv_1']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][0].hist(hsv_g_df_c2[['gv_2']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][0].hist(hsv_g_df_c2[['gv_3']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][0].hist(hsv_g_df_c2[['gv_4']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][1].hist(hsv_g_df_c2[['gv_5']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][1].hist(hsv_g_df_c2[['gv_6']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][1].hist(hsv_g_df_c2[['gv_7']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][1].hist(hsv_g_df_c2[['gv_8']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][1].hist(hsv_g_df_c2[['gv_9']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][2].hist(hsv_g_df_c2[['gv_10']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][2].hist(hsv_g_df_c2[['gv_11']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][2].hist(hsv_g_df_c2[['gv_12']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][2].hist(hsv_g_df_c2[['gv_13']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][2].hist(hsv_g_df_c2[['gv_14']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][3].hist(hsv_g_df_c2[['gv_15']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][3].hist(hsv_g_df_c2[['gv_16']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][3].hist(hsv_g_df_c2[['gv_17']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][3].hist(hsv_g_df_c2[['gv_18']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][3].hist(hsv_g_df_c2[['gv_19']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[0][4].hist(hsv_g_df_c2[['gv_20']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[1][4].hist(hsv_g_df_c2[['gv_21']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[2][4].hist(hsv_g_df_c2[['gv_22']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[3][4].hist(hsv_g_df_c2[['gv_23']].values, range(0, 255, 5), rwidth=0.8, color='red')
axes[4][4].hist(hsv_g_df_c2[['gv_24']].values, range(0, 255, 5), rwidth=0.8, color='red')
# fig.suptitle(f'GammaBase Area Grid: Histogram of V value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_before_cycle_start_V.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_after_cycle_start_V.png'))
fig.suptitle(f'GammaBase Area Grid: Histogram of B value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_before_cycle_start_B.png'))
plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_after_cycle_start_B.png'))
plt.show()
plt.close()


fig, axes = plt.subplots(5, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_g_df_c2[['gh_0']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][0].hist(hsv_g_df_c2[['gh_1']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][0].hist(hsv_g_df_c2[['gh_2']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][0].hist(hsv_g_df_c2[['gh_3']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][0].hist(hsv_g_df_c2[['gh_4']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][1].hist(hsv_g_df_c2[['gh_5']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][1].hist(hsv_g_df_c2[['gh_6']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][1].hist(hsv_g_df_c2[['gh_7']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][1].hist(hsv_g_df_c2[['gh_8']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][1].hist(hsv_g_df_c2[['gh_9']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][2].hist(hsv_g_df_c2[['gh_10']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][2].hist(hsv_g_df_c2[['gh_11']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][2].hist(hsv_g_df_c2[['gh_12']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][2].hist(hsv_g_df_c2[['gh_13']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][2].hist(hsv_g_df_c2[['gh_14']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][3].hist(hsv_g_df_c2[['gh_15']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][3].hist(hsv_g_df_c2[['gh_16']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][3].hist(hsv_g_df_c2[['gh_17']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][3].hist(hsv_g_df_c2[['gh_18']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][3].hist(hsv_g_df_c2[['gh_19']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[0][4].hist(hsv_g_df_c2[['gh_20']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[1][4].hist(hsv_g_df_c2[['gh_21']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[2][4].hist(hsv_g_df_c2[['gh_22']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[3][4].hist(hsv_g_df_c2[['gh_23']].values, range(0, 180, 5), rwidth=0.8, color='blue')
axes[4][4].hist(hsv_g_df_c2[['gh_24']].values, range(0, 180, 5), rwidth=0.8, color='blue')
# fig.suptitle(f'GammaBase Area Grid: Histogram of H value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_before_cycle_start_H.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_after_cycle_start_H.png'))
fig.suptitle(f'GammaBase Area Grid: Histogram of L value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_before_cycle_start_L.png'))
plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_after_cycle_start_L.png'))
plt.show()
plt.close()


fig, axes = plt.subplots(5, 5, figsize=(20, 20), gridspec_kw={"hspace":.0, "wspace":.0}, subplot_kw={"yticks":()})
axes[0][0].hist(hsv_g_df_c2[['gs_0']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][0].hist(hsv_g_df_c2[['gs_1']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][0].hist(hsv_g_df_c2[['gs_2']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][0].hist(hsv_g_df_c2[['gs_3']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][0].hist(hsv_g_df_c2[['gs_4']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][1].hist(hsv_g_df_c2[['gs_5']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][1].hist(hsv_g_df_c2[['gs_6']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][1].hist(hsv_g_df_c2[['gs_7']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][1].hist(hsv_g_df_c2[['gs_8']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][1].hist(hsv_g_df_c2[['gs_9']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][2].hist(hsv_g_df_c2[['gs_10']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][2].hist(hsv_g_df_c2[['gs_11']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][2].hist(hsv_g_df_c2[['gs_12']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][2].hist(hsv_g_df_c2[['gs_13']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][2].hist(hsv_g_df_c2[['gs_14']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][3].hist(hsv_g_df_c2[['gs_15']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][3].hist(hsv_g_df_c2[['gs_16']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][3].hist(hsv_g_df_c2[['gs_17']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][3].hist(hsv_g_df_c2[['gs_18']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][3].hist(hsv_g_df_c2[['gs_19']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[0][4].hist(hsv_g_df_c2[['gs_20']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[1][4].hist(hsv_g_df_c2[['gs_21']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[2][4].hist(hsv_g_df_c2[['gs_22']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[3][4].hist(hsv_g_df_c2[['gs_23']].values, range(0, 255, 5), rwidth=0.8, color='orange')
axes[4][4].hist(hsv_g_df_c2[['gs_24']].values, range(0, 255, 5), rwidth=0.8, color='orange')
# fig.suptitle(f'Keyboard Area Grid: Histogram of S value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_before_cycle_start_S.png'))
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_after_cycle_start_S.png'))
fig.suptitle(f'Keyboard Area Grid: Histogram of A value.png', fontsize=15)
# plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_before_cycle_start_A.png'))
plt.savefig(os.path.join(base_path, '04_output_cycle_extract\\0420pm_GammaBase_after_cycle_start_A.png'))
plt.show()
plt.close()



# -----------------------------------------------------------------------------------------------
# HSV analysis: 0420 pm  check the value and video
# -----------------------------------------------------------------------------------------------

fps = 30

start = 1900
end = 2300

hsv_k_df_c2[(hsv_k_df_c2.sec>=start*fps)&(hsv_k_df_c2.sec<=end*fps)][['sec','kh_9','kv_9']]

rate = 1

fname = time_slots[time_slots.cycid == 0].fname.values[0]
file, ext = os.path.splitext(os.path.basename(fname))
video_file = os.path.join(video_orig_path, str(file[:-7] + '.mp4'))

video = cv2.VideoCapture(video_file)
frame_count = int(video.get(7))
start_count = int(start * fps)
end_count = int(end * fps)

j = start_count
video.set(1, j)

i = 0

while j <= end_count:

    # print(f'cycle: {cyc} - {j}')

    is_read, frame = video.read()
    if not (is_read):
        break
    frame2 = cv2.resize(frame, (frame.shape[1] * rate, frame.shape[0] * rate))
    # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

    clone = frame2.copy()
    cv2.rectangle(clone, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
    cv2.rectangle(clone, (xmin2, ymin2), (xmax2, ymax2), (0, 0, 0), 3)
    cv2.imshow("image", clone)

    grids = grid_k[9]
    h, s, v = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2HSV))

    grids = grid_g[9]
    h2, s2, v2 = cv2.split(cv2.cvtColor(frame2[grids[2]:grids[3], grids[0]:grids[1]], cv2.COLOR_RGB2HSV))

    print(f'Keyboard {round(h.mean())} - {round(s.mean())} - {round(v.mean())} : GammaBase {round(h2.mean())} - {round(s2.mean())} - {round(v2.mean())}')

    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
    i += 1
    j += 1

