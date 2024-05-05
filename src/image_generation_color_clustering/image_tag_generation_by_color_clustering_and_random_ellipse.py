
import os
import glob

import numpy as np
from random import Random

import cv2
import PIL.Image
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

base_path = '/home/kswada/kw/image_processing'


random = Random(1)
random_tag = Random(314)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------------------------------------

def get_bbox_by_inv_maxarea(img, threshold=(250, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, threshold[0], threshold[1], cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    xmin, ymin, width, height = cv2.boundingRect(max_contour)
    return xmin, ymin, width, height, max_contour, contours, gray, bin_img


def get_bbox_by_maxarea(img, threshold=(128, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, threshold[0], threshold[1], cv2.THRESH_BINARY)
    # ----------
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    # ----------
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    xmin, ymin, width, height = cv2.boundingRect(max_contour)
    return xmin, ymin, width, height, max_contour, contours, gray, bin_img


####################################################################################################
# --------------------------------------------------------------------------------------------------
# mask source tag and save
# --------------------------------------------------------------------------------------------------

img_src_tag_dir = os.path.join(base_path, '00_sample_images/tag_image_generation/tag_source')

img_src_tag_path_list = sorted(
    glob.glob(os.path.join(img_src_tag_dir, '*.jpg')) +
    glob.glob(os.path.join(img_src_tag_dir, '*.png'))
)


# ----------
for i in range(len(img_src_tag_path_list)):
    img_tag = cv2.imread(img_src_tag_path_list[i])

    x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_maxarea(img=img_tag, threshold=(128, 255))
    # x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_maxarea(img=img_tag, threshold=(200, 255))

    mask = np.zeros((img_tag.shape[0], img_tag.shape[1], 3))

    # fill-in
    mask = cv2.drawContours(mask, [cont_max], 0, (255, 255, 255), -1).astype('uint8')
    img_tag_fillin = cv2.drawContours(img_tag, [cont_max], 0, (255, 255, 255), -1).astype('uint8')

    # PIL.Image.fromarray(img_tag).show()
    # PIL.Image.fromarray(bin_img).show()
    # PIL.Image.fromarray(mask).show()

    save_path_img = os.path.join(img_src_tag_dir, os.path.basename(img_src_tag_path_list[i]).split('.')[0] + '_fillin.jpg')
    save_path_mask = os.path.join(img_src_tag_dir, os.path.basename(img_src_tag_path_list[i]).split('.')[0] + '_mask.png')
    cv2.imwrite(save_path_img, img_tag_fillin)
    cv2.imwrite(save_path_mask, mask)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# mask source prize and save
# --------------------------------------------------------------------------------------------------

img_src_prize_dir = os.path.join(base_path, '00_sample_images/tag_image_generation/prize_source')

img_src_prize_path_list = sorted(
    glob.glob(os.path.join(img_src_prize_dir, '*.jpg')) +
    glob.glob(os.path.join(img_src_prize_dir, '*.png'))
)


# ----------
for i in range(len(img_src_prize_path_list)):
    img_prize = cv2.imread(img_src_prize_path_list[i])

    x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_inv_maxarea(img=img_prize, threshold=(250, 255))

    mask = np.zeros((img_prize.shape[0], img_prize.shape[1], 3))

    # fill-in
    mask = cv2.drawContours(mask, [cont_max], 0, (255, 255, 255), -1).astype('uint8')
    # img_prize_fillin = cv2.drawContours(img_prize, [cont_max], 0, (255, 255, 255), -1).astype('uint8')

    # PIL.Image.fromarray(img_prize).show()
    # PIL.Image.fromarray(bin_img).show()
    # PIL.Image.fromarray(mask).show()

    save_path_mask = os.path.join(img_src_prize_dir, os.path.basename(img_src_prize_path_list[i]).split('.')[0] + '_mask.png')
    cv2.imwrite(save_path_mask, mask)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# seamless clone of prize image to tag
# --------------------------------------------------------------------------------------------------

prize_name = 'brassband_trading_badge'
img_src_prize_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/prize/{prize_name}/original')
img_src_prize_mask_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/prize/{prize_name}/mask')

tag_name = 'tag02fillin'
img_src_tag_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/tag/{tag_name}/original')
img_src_tag_mask_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/tag/{tag_name}/mask')

img_src_prize_path_list = sorted(
    glob.glob(os.path.join(img_src_prize_dir, '*.jpg')) +
    glob.glob(os.path.join(img_src_prize_dir, '*.png'))
)

img_src_prize_mask_path_list = sorted(
    glob.glob(os.path.join(img_src_prize_mask_dir, '*.jpg')) +
    glob.glob(os.path.join(img_src_prize_mask_dir, '*.png'))
)

img_src_tag_path_list = sorted(
    glob.glob(os.path.join(img_src_tag_dir, '*.jpg')) +
    glob.glob(os.path.join(img_src_tag_dir, '*.png'))
)

img_src_tag_mask_path_list = sorted(
    glob.glob(os.path.join(img_src_tag_mask_dir, '*.jpg')) +
    glob.glob(os.path.join(img_src_tag_mask_dir, '*.png'))
)


i = 0

src = cv2.imread(img_src_prize_path_list[i])
src_mask = cv2.imread(img_src_mask_path_list[i])

tag = cv2.imread(os.path.join(img_src_tag_path_list[0]))
tag_mask = cv2.imread(os.path.join(img_src_tag_mask_path_list[0]))

# PIL.Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)).show()
# PIL.Image.fromarray(cv2.cvtColor(src_mask, cv2.COLOR_BGR2RGB)).show()
# PIL.Image.fromarray(cv2.cvtColor(tag, cv2.COLOR_BGR2RGB)).show()
# PIL.Image.fromarray(cv2.cvtColor(tag_mask, cv2.COLOR_BGR2RGB)).show()

print(f'{tag.shape} : {tag_mask.shape}')
print(f'{src.shape} : {src_mask.shape}')

w_room = 5
h_room = 5
x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_maxarea(img=tag_mask, threshold=(128, 255))
xmin = x - w_room
xmax = x + w + w_room
ymin = y - h_room
ymax = y + h + h_room

tag = tag[ymin:ymax, xmin:xmax]
tag_mask = tag_mask[ymin:ymax, xmin:xmax]
print(tag.shape)


tag_size_min = min(tag_mask.shape[0], tag_mask.shape[1])
src_size_max = max(src.shape[0], src.shape[1])
src_scale = int(tag_size_min / src_size_max) * 0.9

src_resize = cv2.resize(src, (int(src.shape[1] * src_scale), int(src.shape[0] * src_scale)))
src_mask_resize = cv2.resize(src_mask, (int(src_mask.shape[1] * src_scale), int(src_mask.shape[0] * src_scale)))

p = (int(tag.shape[1]/2), int(tag.shape[0]/2))
# p = (100, 100)


comp_min_x = int(tag.shape[1]/2) - int(src_resize.shape[1] / 2)
comp_max_x = comp_min_x + src_resize.shape[1] - 1
comp_min_y = int(tag.shape[0]/2) - int(src_resize.shape[0] / 2)
comp_max_y = comp_min_y + src_resize.shape[0] - 1
print(f'src width  : {comp_max_x - comp_min_x + 1}')
print(f'src height : {comp_max_y - comp_min_y + 1}')
print(f'src resize : {src_resize.shape}')
print(f'src_mask resize : {src_mask_resize.shape}')

tag_src_comp = tag.copy()

tag_src_comp[comp_min_y:comp_max_y + 1, comp_min_x:comp_max_x + 1] = np.where(
    src_mask_resize == 0,
    tag[comp_min_y:comp_max_y + 1, comp_min_x:comp_max_x + 1],
    src_resize)


result1 = cv2.seamlessClone(src_resize, tag, src_mask_resize, p, cv2.NORMAL_CLONE)
result2 = cv2.seamlessClone(src_resize, tag, src_mask_resize, p, cv2.MIXED_CLONE)
result3 = cv2.seamlessClone(src_resize, tag, src_mask_resize, p, cv2.MONOCHROME_TRANSFER)


# ----------
img_to_show = np.hstack([tag_src_comp, result1, result2, result3])
PIL.Image.fromarray(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)).show()


####################################################################################################
# --------------------------------------------------------------------------------------------------
# generate tag with random ellipse by clustered colors of prize
# --------------------------------------------------------------------------------------------------

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def color_clust_and_prob(img_arr, n_clus):
    clt = KMeans(n_clusters=n_clus, n_init='auto')
    clt.fit(img_arr)
    color_clust = clt.cluster_centers_.astype('uint8')
    color_prob = centroid_histogram(clt)
    return clt, color_clust, color_prob


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def add_ellipse(img, num_design,
                color_rep, rep_prob,
                center_w, center_h,
                axes_l_range, axes_s_range, angle_range, startAngle_range, plusAngle_range):
    img_ret = img.copy()
    color_select_list = random_tag.choices(color_rep, k=num_design, weights=rep_prob)
    center_list = [(random_tag.choice(center_w), random_tag.choice(center_h)) for i in range(num_design)]
    axes_list = [(random_tag.randint(axes_l_range[0], axes_l_range[1]), random_tag.randint(axes_s_range[0], axes_s_range[1])) for _ in range(num_design)]
    angle_list = [360 if i % 5 == 0 else random_tag.randint(angle_range[0], angle_range[1]) for i in range(num_design)]
    startAngle_list = [0 if i % 5 == 0 else random_tag.randint(startAngle_range[0], startAngle_range[1]) for i in range(num_design)]
    endAngle_list = [360 if i % 5 == 0 else min(startAngle_list[i] + random_tag.randint(plusAngle_range[0], plusAngle_range[1]), 360) for i in range(num_design)]
    thickness_list = [3 if i % 3 == 0 else -1 for i in range(num_design)]
    # ----------
    for i in range(num_design):
        img_ret = cv2.ellipse(
            img_ret,
            center=center_list[i],
            axes=axes_list[i],
            angle=angle_list[i],
            startAngle=startAngle_list[i],
            endAngle=endAngle_list[i],
            color=(int(color_select_list[i][0]), int(color_select_list[i][1]), int(color_select_list[i][1])),
            thickness=int(thickness_list[i]))
    return img_ret


def add_circle(img, num_design,
               color_rep, rep_prob,
               center_w, center_h,
               radius_range):
    img_ret = img.copy()
    color_select_list = random_tag.choices(color_rep, k=num_design, weights=rep_prob)
    center_list = [(random_tag.choice(center_w), random_tag.choice(center_h)) for i in range(num_design)]
    radius_list = [random_tag.randint(radius_range[0], radius_range[1]) for i in range(num_design)]
    thickness_list = [3 if i % 3 == 0 else -1 for i in range(num_design)]
    # ----------
    for i in range(num_design):
        img_ret = cv2.circle(img_ret,
            center=center_list[i],
            radius=radius_list[i],
            color=(int(color_select_list[i][0]), int(color_select_list[i][1]), int(color_select_list[i][1])),
            thickness=int(thickness_list[i]))
    return img_ret


def add_rectangle(img, num_design,
                  color_rep, rep_prob,
                  start_x, start_y, rec_w_range, rec_h_range):
    img_ret = img.copy()
    color_select_list = random_tag.choices(color_rep, k=num_design, weights=rep_prob)
    start_coord_list = [(random_tag.choice(start_x), random_tag.choice(start_y)) for i in range(num_design)]
    end_coord_list = [(start_coord_list[i][0] + random_tag.randint(rec_w_range[0], rec_w_range[1]),
                       (start_coord_list[i][1] + random_tag.randint(rec_h_range[0], rec_h_range[1]))) for i in range(num_design)]
    thickness_list = [3 if i % 3 == 0 else -1 for i in range(num_design)]
    # ----------
    for i in range(num_design):
        img_ret = cv2.rectangle(img_ret,
            pt1=start_coord_list[i],
            pt2=end_coord_list[i],
            color=(int(color_select_list[i][0]), int(color_select_list[i][1]), int(color_select_list[i][1])),
            thickness=int(thickness_list[i]))
    return img_ret


def add_marker(img, num_design,
                color_rep, rep_prob,
                center_w, center_h,
                markers):
    img_ret = img.copy()
    color_select_list = random_tag.choices(color_rep, k=num_design, weights=rep_prob)
    center_list = [(random_tag.choice(center_w), random_tag.choice(center_h)) for i in range(num_design)]
    marker_list = [random_tag.choice(markers) for i in range(num_design)]
    thickness_list = [random_tag.choice([1,2,3,4,5]) for i in range(num_design)]
    # ----------
    for i in range(num_design):
        img_ret = cv2.drawMarker(
            img_ret,
            position=center_list[i],
            color=(int(color_select_list[i][0]), int(color_select_list[i][1]), int(color_select_list[i][1])),
            markerType=eval(marker_list[i]),
            thickness=int(thickness_list[i]))
    return img_ret


# ----------
tag_group_list = [
    'tag01fillin',
    'tag02fillin',
    'tag03fillin',
    'tag04fillin'
]

prize_group_list = [
    'brassband_trading_badge',
    'seamless_clone',
]

w_room = 5
h_room = 5

n_clus = 7
num_design_ellipse, num_design_circle, num_design_rectangle, num_design_marker = 50, 20, 20, 20
num_vari_ellipse, num_vari_circle, num_vari_rectangle, num_vari_marker = 10, 10, 10, 10

angle_range = (0, 360)
startAngle_range = (0, 360)
plusAngle_range = (0, 360)

markers = [
    "cv2.MARKER_CROSS",
    "cv2.MARKER_TILTED_CROSS",
    "cv2.MARKER_STAR",
    "cv2.MARKER_DIAMOND",
    "cv2.MARKER_SQUARE",
    "cv2.MARKER_TRIANGLE_UP",
    "cv2.MARKER_TRIANGLE_DOWN",
]

img_id = -1
img_id_ellipse, img_id_circle, img_id_rectangle, img_id_marker = -1, -1, -1, -1

save_path = os.path.join(base_path, '00_sample_images', 'tag_image_generation', 'tag', 'tag_all')

for tag_name in tag_group_list:
    img_src_tag_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/tag/{tag_name}/original')
    img_src_tag_mask_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/tag/{tag_name}/mask')
    # ----------
    img_src_tag_path_list = sorted(
        glob.glob(os.path.join(img_src_tag_dir, '*.jpg')) +
        glob.glob(os.path.join(img_src_tag_dir, '*.png'))
    )
    img_src_tag_mask_path_list = sorted(
        glob.glob(os.path.join(img_src_tag_mask_dir, '*.jpg')) +
        glob.glob(os.path.join(img_src_tag_mask_dir, '*.png'))
    )
    tag = cv2.imread(os.path.join(img_src_tag_path_list[0]))
    tag_mask = cv2.imread(os.path.join(img_src_tag_mask_path_list[0]))
    tag_h0, tag_w0, _ = tag.shape
    # ----------
    x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_maxarea(img=tag_mask, threshold=(128, 255))
    xmin = max(x - w_room, 0)
    xmax = min(x + w + w_room, tag_w0)
    ymin = max(y - h_room, 0)
    ymax = min(y + h + h_room, tag_h0)
    # ----------
    tag = tag[ymin:ymax, xmin:xmax]
    tag_bg = np.zeros((tag.shape[0], tag.shape[1], 3))
    tag_mask = tag_mask[ymin:ymax, xmin:xmax]
    tag_h, tag_w, _ = tag.shape
    tag_size_min = min(tag_h, tag_w)
    axes_l_range = (int(tag_size_min * 0.10), int(tag_size_min * 0.25))
    axes_s_range = (int(tag_size_min * 0.05), int(tag_size_min * 0.15))
    radius_range = (int(tag_size_min * 0.05), int(tag_size_min * 0.25))
    center_w = list(range(10, tag_w - 10, int(tag_w / 20)))
    center_h = list(range(10, tag_h - 10, int(tag_h / 20)))
    start_x = list(range(10, tag_w - 10, int(tag_w / 20)))
    start_y = list(range(10, tag_h - 10, int(tag_h / 20)))
    rec_w_range = (int(tag_w * 0.15), int(tag_w * 0.25))
    rec_h_range = (int(tag_h * 0.15), int(tag_h * 0.25))
    # ----------
    for prize_name in prize_group_list:
        img_src_prize_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/prize/{prize_name}/original')
        img_src_prize_mask_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/prize/{prize_name}/mask')
        # ----------
        img_src_prize_path_list = sorted(
            glob.glob(os.path.join(img_src_prize_dir, '*.jpg')) +
            glob.glob(os.path.join(img_src_prize_dir, '*.png'))
        )
        img_src_prize_mask_path_list = sorted(
            glob.glob(os.path.join(img_src_prize_mask_dir, '*.jpg')) +
            glob.glob(os.path.join(img_src_prize_mask_dir, '*.png'))
        )
        # ----------
        for src_idx in range(len(img_src_prize_path_list)):
            src = cv2.imread(img_src_prize_path_list[src_idx])
            src_mask = cv2.imread(img_src_prize_mask_path_list[src_idx])
            src_h0, src_w0, _ = src.shape
            # ----------
            x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_maxarea(img=src_mask, threshold=(128, 255))
            xmin = max(x - w_room, 0)
            xmax = min(x + w + w_room, src_w0)
            ymin = max(y - h_room, 0)
            ymax = min(y + h + h_room, src_h0)
            # ----------
            src = src[ymin:ymax, xmin:xmax]
            src_mask = src_mask[ymin:ymax, xmin:xmax]
            src_h, src_w, _ = src.shape
            src_size_min = min(src_h, src_w)
            # ----------
            src_bg = np.zeros((src.shape[0], src.shape[1], 3))
            src_roi = np.where(src_mask == 0, src_bg, src).astype('uint8')
            # PIL.Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)).show()
            # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
            # ----------
            src_arr = src.reshape((src_roi.shape[0] * src_roi.shape[1], 3))
            # note this is not RGB but BGR
            clt, color_rep, rep_prob = color_clust_and_prob(src_arr, n_clus)
            # ----------
            # show our color bart:  note this is RGB !! not BGR
            # hist = centroid_histogram(clt)
            # bar = plot_colors(hist, clt.cluster_centers_)
            # plt.figure()
            # plt.axis("off")
            # plt.imshow(bar)
            # plt.show()
            # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
            # ----------
            for vari in range(num_vari_ellipse):
                img_id += 1
                img_id_ellipse += 1
                print(f'processing {img_id} : {img_id_ellipse} : {tag_name} : {prize_name} : {src_idx} / {len(img_src_prize_path_list) - 1}')
                tag_color = random_tag.choices(color_rep, k=1)
                tag_copy = np.full((tag_h, tag_w, 3), tag_color)
                tag_ret = add_ellipse(tag_copy, num_design_ellipse, color_rep, rep_prob,
                                center_w, center_h, axes_l_range, axes_s_range, angle_range, startAngle_range, plusAngle_range)
                tag_ret2 = np.where(tag_mask == 0, tag_bg, tag_ret).astype('uint8')
                # PIL.Image.fromarray(cv2.cvtColor(tag, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret2, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
                # ----------
                save_path_img = os.path.join(save_path, 'original', f'IMG_ellipse_{str(img_id_ellipse).zfill(4)}.png')
                save_path_mask = os.path.join(save_path, 'mask', f'IMG_ellipse_{str(img_id_ellipse).zfill(4)}.png')
                cv2.imwrite(save_path_img, tag_ret2)
                cv2.imwrite(save_path_mask, tag_mask)
            for vari in range(num_vari_circle):
                img_id += 1
                img_id_circle += 1
                print(f'processing {img_id} : {img_id_circle} : {tag_name} : {prize_name} : {src_idx} / {len(img_src_prize_path_list) - 1}')
                tag_color = random_tag.choices(color_rep, k=1)
                tag_copy = np.full((tag_h, tag_w, 3), tag_color)
                tag_ret = add_circle(tag_copy, num_design_circle, color_rep, rep_prob, center_w, center_h, radius_range)
                tag_ret2 = np.where(tag_mask == 0, tag_bg, tag_ret).astype('uint8')
                # PIL.Image.fromarray(cv2.cvtColor(tag, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret2, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
                # ----------
                save_path_img = os.path.join(save_path, 'original', f'IMG_circle_{str(img_id_circle).zfill(4)}.png')
                save_path_mask = os.path.join(save_path, 'mask', f'IMG_circle_{str(img_id_circle).zfill(4)}.png')
                cv2.imwrite(save_path_img, tag_ret2)
                cv2.imwrite(save_path_mask, tag_mask)
            for vari in range(num_vari_rectangle):
                img_id += 1
                img_id_rectangle += 1
                print(f'processing {img_id} : {img_id_rectangle} : {tag_name} : {prize_name} : {src_idx} / {len(img_src_prize_path_list) - 1}')
                tag_color = random_tag.choices(color_rep, k=1)
                tag_copy = np.full((tag_h, tag_w, 3), tag_color)
                tag_ret = add_rectangle(tag_copy, num_design_rectangle, color_rep, rep_prob, start_x, start_y, rec_w_range, rec_h_range)
                tag_ret2 = np.where(tag_mask == 0, tag_bg, tag_ret).astype('uint8')
                # PIL.Image.fromarray(cv2.cvtColor(tag, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret2, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
                # ----------
                save_path_img = os.path.join(save_path, 'original', f'IMG_rec_{str(img_id_rectangle).zfill(4)}.png')
                save_path_mask = os.path.join(save_path, 'mask', f'IMG_rec_{str(img_id_rectangle).zfill(4)}.png')
                cv2.imwrite(save_path_img, tag_ret2)
                cv2.imwrite(save_path_mask, tag_mask)
            for vari in range(num_vari_marker):
                img_id += 1
                img_id_marker += 1
                print(f'processing {img_id} : {img_id_marker} : {tag_name} : {prize_name} : {src_idx} / {len(img_src_prize_path_list) - 1}')
                tag_color = random_tag.choices(color_rep, k=1)
                tag_copy = np.full((tag_h, tag_w, 3), tag_color)
                tag_ret = add_marker(tag_copy, num_design_marker, color_rep, rep_prob, center_w, center_h, markers)
                tag_ret2 = np.where(tag_mask == 0, tag_bg, tag_ret).astype('uint8')
                # PIL.Image.fromarray(cv2.cvtColor(tag, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret2, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
                # ----------
                save_path_img = os.path.join(save_path, 'original', f'IMG_marker_{str(img_id_marker).zfill(4)}.png')
                save_path_mask = os.path.join(save_path, 'mask', f'IMG_marker_{str(img_id_marker).zfill(4)}.png')
                cv2.imwrite(save_path_img, tag_ret2)
                cv2.imwrite(save_path_mask, tag_mask)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# generate tag with random grid circle and rectangle by clustered colors of prize
# --------------------------------------------------------------------------------------------------

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def color_clust_and_prob(img_arr, n_clus):
    clt = KMeans(n_clusters=n_clus, n_init='auto')
    clt.fit(img_arr)
    color_clust = clt.cluster_centers_.astype('uint8')
    color_prob = centroid_histogram(clt)
    return clt, color_clust, color_prob


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def add_circle(img, num_design,
               color_rep, rep_prob,
               center_w, center_h,
               radius_range):
    img_ret = img.copy()
    color_select_list = random_tag.choices(color_rep, k=num_design, weights=rep_prob)
    center_list = [(random_tag.choice(center_w), random_tag.choice(center_h)) for i in range(num_design)]
    radius_list = [random_tag.randint(radius_range[0], radius_range[1]) for i in range(num_design)]
    # thickness_list = [3 if i % 3 == 0 else -1 for i in range(num_design)]
    thickness_list = [-1 for i in range(num_design)]
    # ----------
    for i in range(num_design):
        img_ret = cv2.circle(img_ret,
            center=center_list[i],
            radius=radius_list[i],
            color=(int(color_select_list[i][0]), int(color_select_list[i][1]), int(color_select_list[i][1])),
            thickness=int(thickness_list[i]))
    return img_ret


def add_rectangle(img, num_design,
                  color_rep, rep_prob,
                  start_x, start_y, rec_w_range, rec_h_range):
    img_ret = img.copy()
    color_select_list = random_tag.choices(color_rep, k=num_design, weights=rep_prob)
    start_coord_list = [(random_tag.choice(start_x), random_tag.choice(start_y)) for i in range(num_design)]
    end_coord_list = [(start_coord_list[i][0] + random_tag.randint(rec_w_range[0], rec_w_range[1]),
                       (start_coord_list[i][1] + random_tag.randint(rec_h_range[0], rec_h_range[1]))) for i in range(num_design)]
    # thickness_list = [3 if i % 3 == 0 else -1 for i in range(num_design)]
    thickness_list = [-1 for i in range(num_design)]
    # ----------
    for i in range(num_design):
        img_ret = cv2.rectangle(img_ret,
            pt1=start_coord_list[i],
            pt2=end_coord_list[i],
            color=(int(color_select_list[i][0]), int(color_select_list[i][1]), int(color_select_list[i][1])),
            thickness=int(thickness_list[i]))
    return img_ret


# ----------
tag_group_list = [
    'tag01fillin',
    'tag02fillin',
    'tag03fillin',
    'tag04fillin'
]

prize_group_list = [
    'brassband_trading_badge',
    'seamless_clone',
]

w_room = 5
h_room = 5

n_clus = 5
num_design_circle, num_design_rectangle = 200, 200
num_vari_circle, num_vari_rectangle = 10, 10


img_id = -1
img_id_circle, img_id_rectangle = -1, -1

save_path = os.path.join(base_path, '00_sample_images', 'tag_image_generation', 'tag_all2')


# ----------
for prize_name in prize_group_list:
    img_src_prize_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/prize/{prize_name}/original')
    img_src_prize_mask_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/prize/{prize_name}/mask')
    # ----------
    img_src_prize_path_list = sorted(
        glob.glob(os.path.join(img_src_prize_dir, '*.jpg')) +
        glob.glob(os.path.join(img_src_prize_dir, '*.png'))
    )
    img_src_prize_mask_path_list = sorted(
        glob.glob(os.path.join(img_src_prize_mask_dir, '*.jpg')) +
        glob.glob(os.path.join(img_src_prize_mask_dir, '*.png'))
    )
    for src_idx in range(len(img_src_prize_path_list)):
        src = cv2.imread(img_src_prize_path_list[src_idx])
        src_mask = cv2.imread(img_src_prize_mask_path_list[src_idx])
        src_h0, src_w0, _ = src.shape
        # ----------
        x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_maxarea(img=src_mask, threshold=(128, 255))
        xmin = max(x - w_room, 0)
        xmax = min(x + w + w_room, src_w0)
        ymin = max(y - h_room, 0)
        ymax = min(y + h + h_room, src_h0)
        # ----------
        src = src[ymin:ymax, xmin:xmax]
        src_mask = src_mask[ymin:ymax, xmin:xmax]
        src_h, src_w, _ = src.shape
        src_size_min = min(src_h, src_w)
        # ----------
        src_bg = np.zeros((src.shape[0], src.shape[1], 3))
        src_roi = np.where(src_mask == 0, src_bg, src).astype('uint8')
        # PIL.Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)).show()
        # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
        # ----------
        # KMeans ++
        # src_arr = src.reshape((src_roi.shape[0] * src_roi.shape[1], 3))
        # # note this is not RGB but BGR
        # # clt, color_rep, rep_prob = color_clust_and_prob(src_arr, n_clus)
        # clt = KMeans(n_clusters=n_clus, n_init='auto')
        # clt.fit(src_arr)
        # color_rep = clt.cluster_centers_.astype('uint8')
        # numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        # (rep_prob, _) = np.histogram(clt.labels_, bins=numLabels)
        # rep_prob = rep_prob.astype("float")
        # rep_prob /= rep_prob.sum()
        # ----------
        src_arr = src.reshape((src_roi.shape[0] * src_roi.shape[1], 3))
        gm = GaussianMixture(n_components=n_clus, random_state=0).fit(src_arr)
        color_rep = gm.means_.astype('uint8')
        labels_gm = gm.predict(src_arr)
        numLabels = np.arange(0, len(np.unique(labels_gm)) + 1)
        (rep_prob, _) = np.histogram(labels_gm, bins=numLabels)
        rep_prob = rep_prob.astype("float")
        rep_prob /= rep_prob.sum()
        # ----------
        # show our color bart:  note this is RGB !! not BGR
        # hist = centroid_histogram(clt)
        # bar = plot_colors(hist, clt.cluster_centers_)
        # plt.figure()
        # plt.axis("off")
        # plt.imshow(bar)
        # plt.show()
        # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
        # ----------
        for tag_name in tag_group_list:
            img_src_tag_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/tag/{tag_name}/original')
            img_src_tag_mask_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/tag/{tag_name}/mask')
            # ----------
            img_src_tag_path_list = sorted(
                glob.glob(os.path.join(img_src_tag_dir, '*.jpg')) +
                glob.glob(os.path.join(img_src_tag_dir, '*.png'))
            )
            img_src_tag_mask_path_list = sorted(
                glob.glob(os.path.join(img_src_tag_mask_dir, '*.jpg')) +
                glob.glob(os.path.join(img_src_tag_mask_dir, '*.png'))
            )
            tag = cv2.imread(os.path.join(img_src_tag_path_list[0]))
            tag_mask = cv2.imread(os.path.join(img_src_tag_mask_path_list[0]))
            tag_h0, tag_w0, _ = tag.shape
            # ----------
            x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_maxarea(img=tag_mask, threshold=(128, 255))
            xmin = max(x - w_room, 0)
            xmax = min(x + w + w_room, tag_w0)
            ymin = max(y - h_room, 0)
            ymax = min(y + h + h_room, tag_h0)
            # ----------
            tag = tag[ymin:ymax, xmin:xmax]
            tag_bg = np.zeros((tag.shape[0], tag.shape[1], 3))
            tag_mask = tag_mask[ymin:ymax, xmin:xmax]
            tag_h, tag_w, _ = tag.shape
            tag_size_min = min(tag_h, tag_w)
            # ----------
            # radius_range = (int(tag_size_min * 0.05), int(tag_size_min * 0.25))
            radius_range = (int(tag_size_min * 0.15), int(tag_size_min * 0.15))
            # ----------
            center_w = list(range(0, tag_w, radius_range[0]*2))
            center_h = list(range(0, tag_h, radius_range[0]*2))
            start_x = list(range(0, tag_w, radius_range[0]*2))
            start_y = list(range(0, tag_h, radius_range[0]*2))
            # rec_w_range = (int(tag_w * 0.15), int(tag_w * 0.15))
            # rec_h_range = (int(tag_h * 0.15), int(tag_h * 0.15))
            rec_w_range = (radius_range[0] * 2, radius_range[0] * 2)
            rec_h_range = (radius_range[0] * 2, radius_range[0] * 2)
            # ----------
            for vari in range(num_vari_circle):
                img_id += 1
                img_id_circle += 1
                print(f'processing {img_id} : {img_id_circle} : {tag_name} : {prize_name} : {src_idx} / {len(img_src_prize_path_list) - 1}')
                # ----------
                # tag_color = random_tag.choices(color_rep, k=1)
                tag_color = color_rep[np.argmax(rep_prob)]
                # ----------
                tag_copy = np.full((tag_h, tag_w, 3), tag_color)
                tag_ret = add_circle(tag_copy, num_design_circle, color_rep, rep_prob, center_w, center_h, radius_range)
                tag_ret2 = np.where(tag_mask == 0, tag_bg, tag_ret).astype('uint8')
                # PIL.Image.fromarray(cv2.cvtColor(tag, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret2, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
                # ----------
                save_path_img = os.path.join(save_path, tag_name, 'original', f'IMG_circle_{prize_name}_{str(img_id_circle).zfill(4)}.png')
                save_path_mask = os.path.join(save_path, tag_name, 'mask', f'IMG_circle_{prize_name}_{str(img_id_circle).zfill(4)}.png')
                cv2.imwrite(save_path_img, tag_ret2)
                cv2.imwrite(save_path_mask, tag_mask)
            for vari in range(num_vari_rectangle):
                img_id += 1
                img_id_rectangle += 1
                print(f'processing {img_id} : {img_id_rectangle} : {tag_name} : {prize_name} : {src_idx} / {len(img_src_prize_path_list) - 1}')
                # ----------
                # tag_color = random_tag.choices(color_rep, k=1)
                tag_color = color_rep[np.argmax(rep_prob)]
                # ----------
                tag_copy = np.full((tag_h, tag_w, 3), tag_color)
                tag_ret = add_rectangle(tag_copy, num_design_rectangle, color_rep, rep_prob, start_x, start_y, rec_w_range, rec_h_range)
                tag_ret2 = np.where(tag_mask == 0, tag_bg, tag_ret).astype('uint8')
                # PIL.Image.fromarray(cv2.cvtColor(tag, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(tag_ret2, cv2.COLOR_BGR2RGB)).show()
                # PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()
                # ----------
                save_path_img = os.path.join(save_path, tag_name, 'original', f'IMG_rec_{prize_name}_{str(img_id_rectangle).zfill(4)}.png')
                save_path_mask = os.path.join(save_path, tag_name, 'mask', f'IMG_rec_{prize_name}_{str(img_id_rectangle).zfill(4)}.png')
                cv2.imwrite(save_path_img, tag_ret2)
                cv2.imwrite(save_path_mask, tag_mask)


####################################################################################################
# --------------------------------------------------------------------------------------------------
# check clustering
# --------------------------------------------------------------------------------------------------

prize_group_list = [
    'brassband_trading_badge',
    'seamless_clone',
]

prize_name = prize_group_list[0]

img_src_prize_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/prize/{prize_name}/original')
img_src_prize_mask_dir = os.path.join(base_path, f'00_sample_images/tag_image_generation/prize/{prize_name}/mask')

img_src_prize_path_list = sorted(
    glob.glob(os.path.join(img_src_prize_dir, '*.jpg')) +
    glob.glob(os.path.join(img_src_prize_dir, '*.png'))
)

img_src_prize_mask_path_list = sorted(
    glob.glob(os.path.join(img_src_prize_mask_dir, '*.jpg')) +
    glob.glob(os.path.join(img_src_prize_mask_dir, '*.png'))
)

src_idx = 1

src = cv2.imread(img_src_prize_path_list[src_idx])
src_mask = cv2.imread(img_src_prize_mask_path_list[src_idx])

src_h0, src_w0, _ = src.shape

x, y, w, h, cont_max, conts, gray, bin_img = get_bbox_by_maxarea(img=src_mask, threshold=(128, 255))
xmin = max(x - w_room, 0)
xmax = min(x + w + w_room, src_w0)
ymin = max(y - h_room, 0)
ymax = min(y + h + h_room, src_h0)

src = src[ymin:ymax, xmin:xmax]
src_mask = src_mask[ymin:ymax, xmin:xmax]
src_h, src_w, _ = src.shape
src_size_min = min(src_h, src_w)

src_bg = np.zeros((src.shape[0], src.shape[1], 3))
src_roi = np.where(src_mask == 0, src_bg, src).astype('uint8')

# PIL.Image.fromarray(cv2.cvtColor(src_roi, cv2.COLOR_BGR2RGB)).show()


# ----------
n_clus = 5

src_arr = src.reshape((src_roi.shape[0] * src_roi.shape[1], 3))


gm = GaussianMixture(n_components=n_clus, random_state=0).fit(src_arr)
clt = KMeans(n_clusters=n_clus, n_init='auto')
clt.fit(src_arr)


# ----------
color_rep_km = clt.cluster_centers_.astype('uint8')
color_rep_gm = gm.means_.astype('uint8')

labels_km = clt.labels_
labels_gm = gm.predict(src_arr)

numLabels_km = np.arange(0, len(np.unique(labels_km)) + 1)
numLabels_gm = np.arange(0, len(np.unique(labels_gm)) + 1)

(rep_prob_km, _) = np.histogram(labels_km, bins=numLabels_km)
(rep_prob_gm, _) = np.histogram(labels_gm, bins=numLabels_gm)

rep_prob_km = rep_prob_km.astype("float")
rep_prob_km /= rep_prob_km.sum()

rep_prob_gm = rep_prob_gm.astype("float")
rep_prob_gm /= rep_prob_gm.sum()


bar_km = np.zeros((50, 300, 3), dtype="uint8")
bar_gm = np.zeros((50, 300, 3), dtype="uint8")

startX = 0
for (percent, color) in zip(rep_prob_km, color_rep_km):
    endX = startX + (percent * 300)
    cv2.rectangle(bar_km, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
    startX = endX

startX = 0
for (percent, color) in zip(rep_prob_gm, color_rep_gm):
    endX = startX + (percent * 300)
    cv2.rectangle(bar_gm, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
    startX = endX

PIL.Image.fromarray(cv2.cvtColor(bar_km, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(bar_gm, cv2.COLOR_BGR2RGB)).show()
