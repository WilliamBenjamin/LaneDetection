import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
from moviepy.editor import VideoFileClip
import math

def show_images(images, cmap = None):
    cols = 2
    rows = (len(images)+1)/cols
    plt.figure(figsize=(20,10))
    for i, image in enumerate(images):
        image.shape
        plt.subplot(rows,cols,i+1)
        if len(image.shape) > 2:
            cmap = 'BrBG'
        else:
            cmap='gray'
        plt.imshow(image,cmap)

    plt.show()

def bgr2rgb(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def select_color_RGB(image):
    # white color mask
    lower = np.uint8([  200, 200,   200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([0, 190,   190])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image,image, mask = mask)
    return masked_image


def select_color_HSV(image):
    # white color mask
    image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower = np.uint8([  0, 0,   220])
    upper = np.uint8([255, 50, 255])
    white_mask = cv2.inRange(image_hsv, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   100, 0])
    upper = np.uint8([ 52, 255, 255])
    yellow_mask = cv2.inRange(image_hsv, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked_image = cv2.bitwise_and(image,image, mask = mask)
    return masked_image


def region_select(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    vertices = np.array( [[[0.25*(xsize),0.8*(ysize)],[0.75*(xsize),0.8*(ysize)],[0.55*(xsize),0.60*(ysize)],[0.45*(xsize),0.60*(ysize)]]], dtype=np.int32 )
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def canny_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    low_threshold = 150
    high_threshold = 180
    canny = cv2.Canny(blur_gray, low_threshold, high_threshold)
    return canny

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines2(img, lines, color=[255, 0, 0], thickness=3):
    for x1,y1,x2,y2 in lines:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)




def hough_lines(img):
    min_line_len = 5
    max_line_gap = 10
    rho = 1
    theta = np.pi/180
    threshold = 10
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def get_left_right_lanes(lines):  # get the left and right lanes slope and intercept
    lines_l = []
    lines_r = []
    weights_l = []
    weights_r = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                lines_l.append((slope, intercept))
                weights_l.append((length))
            else:
                lines_r.append((slope, intercept))
                weights_r.append((length))

    lane_left = np.dot(weights_l, lines_l) / np.sum(weights_l) if len(weights_l) > 0 else None
    lane_right = np.dot(weights_r, lines_r) / np.sum(weights_r) if len(weights_r) > 0 else None

    return lane_left, lane_right


def get_image_points_from_slope_intercept(lines, image):
    ysize = image.shape[0]
    xsize = image.shape[1]

    line_array = np.asanyarray(lines)
    lane_points = []
    y1 = 0.9 * ysize
    y2 = 0.7 * ysize
    for slope, intercept in line_array:
        print(slope)
        x1 = (y1 - intercept) / slope
        x1_int = math.floor(x1)
        x2 = (y2 - intercept) / slope
        x2_int = math.floor(x2)

        y1_int = math.floor(y1)
        y2_int = math.floor(y2)
        lane_points.append((x1_int, y1_int, x2_int, y2_int))

    return lane_points

def process_image(image): # takes and RGB image and returns RGB image with lane lines
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    color_masked_image = select_color_HSV(image)
    region_masked_image = region_select(color_masked_image)
    edges_image = canny_edges(region_masked_image)
    #hough_lines_image = hough_lines(edges_image)

    min_line_len = 5
    max_line_gap = 10
    rho = 1
    theta = np.pi / 180
    threshold = 10
    lines = cv2.HoughLinesP(edges_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    lanes = get_left_right_lanes(lines)
    ll = get_image_points_from_slope_intercept(lanes, image)
    gg = np.asanyarray(ll)
    draw_lines2(image, gg)
    image = bgr2rgb(image)
    return image

image = cv2.imread('./test_images/tchallenge2.jpg')
op_img = process_image(image)
plt.imshow(op_img)



clip_output_path = './output/solidWhiteRight.mp4'
clip = VideoFileClip('./test_videos/solidWhiteRight.mp4') #.subclip(0,5)
new_clip = clip.fl_image(process_image)
new_clip.write_videofile(clip_output_path, audio=False)


