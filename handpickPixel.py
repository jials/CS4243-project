import cv2
import numpy as np

selected_pixels = []
original_image = []
image = []
def mouseClick(event, x, y, flags, param):
    if event == 1: #mouse click
        global image, selected_pixels
        image = markImageAtPoint(image, y, x, 9)
        selected_pixels.append([x, y])

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
