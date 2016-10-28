import cv2
import numpy as np
import imageMarker

selected_pixels = []
original_image = []
image = []
def mouseClick(event, x, y, flags, param):
    if event == 1: #mouse click
        global image, selected_pixels
        image = imageMarker.markImageAtPoint(image, y, x, 9)
        selected_pixels.append([x, y])

def handpickImage(img):
    global image
    original_image = img
    image = img.copy()
    cv2.namedWindow('first frame')
    cv2.setMouseCallback('first frame', mouseClick)
    i = 0;
    while True:
        cv2.imshow('first frame', image)
        key = cv2.waitKey(1) & 0xFF
        i = i + 1

        if key == ord('r'):
            selected_pixels = []
            image = original_image.copy()

        if key == ord('c'):
            break

    cv2.destroyAllWindows()

    global selected_pixels
    return selected_pixels
