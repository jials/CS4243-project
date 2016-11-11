import cv2
import numpy as np
import imageMarker
import convolution
import os
import util

def mark_corners_on_all_images(images, folder_name):
    marked_images = []
    marked_frame_coordinates = []
    for i in range(len(images)):
    # for i in range(3):
        print('frame', i, 'out of', len(images))
        image_name = os.path.join(folder_name, 'corners', 'frame' + str(i) + '.jpg')
        marked_image, marked_coordinates = markCornerOnImage(images[i], image_name)
        marked_frame_coordinates.append(marked_coordinates)
        marked_images.append(marked_image)
    return marked_images

def video_to_corner_detection(video_file):
    video_images, fps = util.get_all_frame_images_and_fps(video_file)
    video_file_name, _ = video_file.split('.')
    if not os.path.isdir('./' + video_file_name + '/corners'):
        os.mkdir(video_file_name + '/corners')
    marked_images = mark_corners_on_all_images(video_images, video_file_name)
    return marked_images, fps

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

def get_response_matrix(W_xx, W_xy, W_yy, interval, image_shape):
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

def mark_corner(img, responseMatrix, maxResponse, interval):
    rows = []
    cols = []
    marked_coordinates = []
    for y_offset in range(interval, len(responseMatrix), interval):
        for x_offset in range(interval, len(responseMatrix[0]), interval):
            currentResponse = responseMatrix[y_offset][x_offset]
            if currentResponse > 0.1 * maxResponse:
                img = imageMarker.mark_image_at_point(img, y_offset, x_offset, 15) #size of 9 not as obvious
                marked_coordinates.append([x_offset, y_offset])
    return img, marked_coordinates

def markCornerOnImage(image, image_name):
    color_img = image
    gs_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gx = np.absolute(convolution.get_sobel_horizontal_edge_strength(gs_img))
    gy = np.absolute(convolution.get_sobel_vertical_edge_strength(gs_img))
    I_xx = gx * gx
    I_xy = gx * gy
    I_yy = gy * gy
    W_xx = convolution.convolve(I_xx, gauss_kernels(3))
    W_xy = convolution.convolve(I_xy, gauss_kernels(3))
    W_yy = convolution.convolve(I_yy, gauss_kernels(3))
    interval = 10
    responseMatrix, maxResponse = get_response_matrix(W_xx, W_xy, W_yy, interval, gs_img.shape)
    marked_img, marked_coordinates = mark_corner(color_img, responseMatrix, maxResponse, interval)
    cv2.imwrite(image_name, marked_img)
    return marked_img, marked_coordinates
