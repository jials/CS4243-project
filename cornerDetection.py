import cv2
import numpy as np
import imageMarker
import convolution
import os

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

def markCorner(img, responseMatrix, maxResponse, interval):
    rows = []
    cols = []
    marked_coordinates = []
    for y_offset in range(interval, len(responseMatrix), interval):
        for x_offset in range(interval, len(responseMatrix[0]), interval):
            currentResponse = responseMatrix[y_offset][x_offset]
            if currentResponse > 0.1 * maxResponse:
                img = imageMarker.markImageAtPoint(img, y_offset, x_offset, 15) #size of 9 not as obvious
                marked_coordinates.append([x_offset, y_offset])
    return img, marked_coordinates

def markCornerOnImage(image, image_name):
    color_img = image
    gs_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gx = np.absolute(convolution.getSobelHorizontalEdgeStrength(gs_img))
    gy = np.absolute(convolution.getSobelVerticalEdgeStrength(gs_img))
    I_xx = gx * gx
    I_xy = gx * gy
    I_yy = gy * gy
    W_xx = convolution.convolve(I_xx, gauss_kernels(3))
    W_xy = convolution.convolve(I_xy, gauss_kernels(3))
    W_yy = convolution.convolve(I_yy, gauss_kernels(3))
    interval = 10
    responseMatrix, maxResponse = getResponseMatrix(W_xx, W_xy, W_yy, interval, gs_img.shape)
    marked_img, marked_coordinates = markCorner(color_img, responseMatrix, maxResponse, interval)
    cv2.imwrite(image_name, marked_img)
    return marked_img, marked_coordinates
