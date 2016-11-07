import cv2
import numpy as np
import imageMarker

red = np.uint8([0, 0, 255])
orange = np.uint8([0, 127, 255])
yellow = np.uint8([0, 255, 255])
green = np.uint8([0, 255, 0])
blue = np.uint8([255, 0, 0])
colors = [red, orange, yellow, green, blue]

selected_pixels = []
original_image = []
image = []
editing_index = -1

def mark_all_points():
    marked_image = original_image.copy()
    for index in range(len(selected_pixels)):
        pixel = selected_pixels[index]
        x = int(pixel[0])
        y = int(pixel[1])
        marked_image = imageMarker.mark_image_at_point(marked_image, y, x, 9, colors[index])
    return marked_image

def mouse_click(event, x, y, flags, param):
    global image, selected_pixels, editing_index
    if event == 1: #mouse click
        for index in range(len(selected_pixels)):
            pixel = selected_pixels[index]
            selected_x = pixel[0]
            selected_y = pixel[1]
            if selected_x - 4 <= x and x <= selected_x + 4 and selected_y - 4 <= y and y <= selected_y + 4:
                editing_index = index
                return

        if len(selected_pixels) < len(colors):
            selected_pixels.append([x, y])
            image = mark_all_points()
    elif event == 4: #mouse up
        if editing_index > -1:
            selected_pixels[editing_index] = [x, y]
            image = mark_all_points()
            editing_index = -1


def handpick_image(img, estimated_pixels = []):
    global image, original_image, selected_pixels

    selected_pixels = estimated_pixels
    original_image = img
    image = img.copy()
    if len(estimated_pixels) > 0:
        image = mark_all_points()
    cv2.namedWindow('pick point')
    cv2.setMouseCallback('pick point', mouse_click)
    while True:
        cv2.imshow('pick point', image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            selected_pixels = []
            image = original_image.copy()

        if key == ord('c'):
            break

    # cv2.destroyAllWindows()

    return selected_pixels
