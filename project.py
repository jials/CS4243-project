import cv2
import cv2.cv as cv
import numpy as np
import sys
import os
import imagesToVideo
from math import sqrt

def getStrength(horizontal_sum, vertical_sum):
    return sqrt(horizontal_sum * horizontal_sum + vertical_sum * vertical_sum)

def MyConvolve(img, ff):
    ff = np.flipud(ff)
    result = np.zeros(img.shape)
    for y_offset in range(len(img) - 2):
        for x_offset in range(len(img[0]) - 2):
            temp_arr = [];
            for row in img[y_offset : 3 + y_offset]:
                temp_arr.append(row[x_offset : 3 + x_offset])
            filtered_arr = np.array(temp_arr) * ff
            filtered_sum = sum(map(sum, filtered_arr))
            result[y_offset + 1][x_offset + 1] = filtered_sum
    return result

def getSobelHorizontalEdgeStrength(img):
    horizontal_sobel_filter = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return MyConvolve(img, horizontal_sobel_filter)

def getSobelVerticalEdgeStrength(img):
    vertical_sobel_filter = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    return MyConvolve(img, vertical_sobel_filter)

def getSobelEdge(img):
    horizontal_sobel_edge_strength = np.absolute(getSobelHorizontalEdgeStrength(img))
    vertical_sobel_edge_strength = np.absolute(getSobelVerticalEdgeStrength(img))

    result = np.zeros(img.shape)
    maxStr = 0
    minStr = 99999
    for y in range(1, len(img) - 1):
        for x in range(1, len(img[0]) - 1):
            strength= getStrength(horizontal_sobel_edge_strength[y][x], vertical_sobel_edge_strength[y][x])
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


def videoToSobelEdgeDetection(video_file_name, extension):
    video_images, fps = getAllFrameImagesAndFps(video_file_name, extension)
    if not os.path.isdir('./' + video_file_name):
        os.mkdir(video_file_name)
    if not os.path.isdir('./' + video_file_name + '/edge'):
        os.mkdir(video_file_name + '/edge')

    images = []
    for fr in range(len(video_images)):
    # for fr in range(10):
        print('frame', fr, 'out of', len(video_images))
        img = video_images[fr]
        gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = getSobelEdge(gs_img)
        images.append(result);
        image_name = os.path.join(video_file_name, 'edge', 'frame' + str(fr) + '.jpg')
        cv2.imwrite(image_name, result);
    return images, fps

def getAllFrameImagesAndFps(video_file_name, extension):
    cap = cv2.VideoCapture(video_file_name + extension)
    frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CV_CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

    print('Frame Width ', frame_width)
    print('Frame Height ', frame_height)
    print('FPS ', fps)
    print('Frame Count', frame_count)

    images = []
    for fr in range(0,frame_count):
        _,img = cap.read()
        images.append(img)
    cap.release()
    return images, fps;

def gauss_kernels(size,sigma=1.0):
    ## returns a 2d gaussian kernel
    if size<3:
        size = 3
    m = size/2
    x, y = np.mgrid[-m:m+1, -m:m+1]
    kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    kernel_sum = kernel.sum()
    if not sum==0:
        kernel = kernel/kernel_sum
    return kernel

def getResponseMatrix(W_xx, W_xy, W_yy, interval, image_shape):
    k = 0.06
    responseMatrix = np.zeros(image_shape)
    maxResponse = 0
    for y_offset in range(interval, len(W_xx), interval):
        for x_offset in range(interval, len(W_xx[0]), interval):
            W_xx_element = W_xx[y_offset][x_offset]
            W_xy_element = W_xy[y_offset][x_offset]
            W_yy_element = W_yy[y_offset][x_offset]
            W = np.matrix([[W_xx_element, W_xy_element],[W_xy_element, W_yy_element]])
            detW = np.linalg.det(W)
            traceW = np.trace(W)
            response = detW - k * traceW * traceW
            responseMatrix[y_offset][x_offset] = response
            if response > maxResponse:
                maxResponse = response
    return responseMatrix, maxResponse

def markImageAtPoint(img, y, x, size):
    blue = np.uint8([255, 128, 0])
    if size < 1:
        size = 1
    negative_offset = (size - 1) / 2
    positive_offset = size / 2 + 1

    min_y = max(0, y - negative_offset)
    max_y = min(img.shape[0], y + positive_offset)
    min_x = max(0, x - negative_offset)
    max_x = min(img.shape[1], x + positive_offset)
    for y_offset in range(min_y, max_y):
        img[y_offset][min_x] = blue
        img[y_offset][min_x + 1] = blue
        img[y_offset][max_x - 1] = blue
        img[y_offset][max_x - 2] = blue

    for x_offset in range(min_x, max_x):
        img[min_y][x_offset] = blue
        img[min_y + 1][x_offset] = blue
        img[max_y - 1][x_offset] = blue
        img[max_y - 2][x_offset] = blue

    return img

def markCorner(img, responseMatrix, maxResponse, interval):
    rows = []
    cols = []
    for y_offset in range(interval, len(responseMatrix), interval):
        for x_offset in range(interval, len(responseMatrix[0]), interval):
            currentResponse = responseMatrix[y_offset][x_offset]
            if currentResponse > 0.1 * maxResponse:
                img = markImageAtPoint(img, y_offset, x_offset, 15) #size of 9 not as obvious
    return img

def videoToCornerDetection(video_file_name, extension):
    video_images, fps = getAllFrameImagesAndFps(video_file_name, extension)
    if not os.path.isdir('./' + video_file_name):
        os.mkdir(video_file_name)
    if not os.path.isdir('./' + video_file_name + '/corners'):
        os.mkdir(video_file_name + '/corners')

    images = [];
    for i in range(len(video_images)):
    # for i in range(10):
        print('frame', i, 'out of', len(video_images))
        gs_img = cv2.cvtColor(video_images[i], cv2.COLOR_BGR2GRAY)
        gx = np.absolute(getSobelHorizontalEdgeStrength(gs_img))
        gy = np.absolute(getSobelVerticalEdgeStrength(gs_img))
        I_xx = gx * gx
        I_xy = gx * gy
        I_yy = gy * gy
        W_xx = MyConvolve(I_xx, gauss_kernels(3))
        W_xy = MyConvolve(I_xy, gauss_kernels(3))
        W_yy = MyConvolve(I_yy, gauss_kernels(3))
        interval = 10
        responseMatrix, maxResponse = getResponseMatrix(W_xx, W_xy, W_yy, interval, gs_img.shape)
        imgColor = video_images[i]
        markedImg = markCorner(imgColor, responseMatrix, maxResponse, interval)
        images.append(markedImg)
        image_name = os.path.join(video_file_name, 'corners', 'frame' + str(i) + '.jpg')
        cv2.imwrite(image_name, markedImg)
    return images, fps

arguments = sys.argv
if len(arguments) > 1:
    task = arguments[1]
    if task == 'edge' and len(arguments) == 4:
        video_file_name = arguments[2]
        extension = arguments[3]
        images, fps = videoToSobelEdgeDetection(video_file_name, extension)
        video_path = os.path.join(video_file_name, video_file_name + '_sobel_edge')
        imagesToVideo.convertGrayscaleImagesToVideo(images, fps, video_path)
    elif task == 'corner' and len(arguments) == 4:
        video_file_name = arguments[2]
        extension = arguments[3]
        images, fps = videoToCornerDetection(video_file_name, extension)
        video_path = os.path.join(video_file_name, video_file_name + '_corner')
        imagesToVideo.convertImagesToVideo(images, fps, video_path)
