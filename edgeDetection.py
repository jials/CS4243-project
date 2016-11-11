import cv2
import numpy as np
import convolution
from math import sqrt
import os
import util

def video_to_sobel_edge_detection(video_file):
    video_images, fps = util.get_all_frame_images_and_fps(video_file)
    video_file_name, _ = video_file.split('.')
    if not os.path.isdir('./' + video_file_name + '/edge'):
        os.mkdir(video_file_name + '/edge')

    edge_images = detect_edges(video_images, video_file_name)
    return edge_images, fps

def get_strength(horizontal_sum, vertical_sum):
    return sqrt(horizontal_sum * horizontal_sum + vertical_sum * vertical_sum)

def get_sobel_edge(img):
    horizontal_sobel_edge_strength = np.absolute(convolution.get_sobel_horizontal_edge_strength(img))
    vertical_sobel_edge_strength = np.absolute(convolution.get_sobel_vertical_edge_strength(img))

    result = np.zeros(img.shape)
    maxStr = 0
    minStr = 99999
    for y in range(1, len(img) - 1):
        for x in range(1, len(img[0]) - 1):
            strength= get_strength(horizontal_sobel_edge_strength[y][x], vertical_sobel_edge_strength[y][x])
            result[y][x] = strength
            if (strength > maxStr):
                maxStr = strength
            elif (strength < minStr):
                minStr = strength

    strengthRange = maxStr - minStr
    for y in range(1, len(img) - 1):
        for x in range(1, len(img[0]) - 1):
            result[y][x] = (result[y][x] - minStr) / strengthRange * 255
    return result

def detect_edges(images, folder_name):
    edge_images = []
    for fr in range(len(images)):
    # for fr in range(10):
        print('frame', fr, 'out of', len(images))
        img = images[fr]
        gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = get_sobel_edge(gs_img)
        edge_images.append(result)
        image_name = os.path.join(folder_name, 'edge', 'frame' + str(fr) + '.jpg')
        cv2.imwrite(image_name, result)
    return edge_images
