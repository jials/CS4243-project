import cv2
import numpy as np
import imageMarker

selected_pixels = []
original_image = []
image = []
editing_index = -1
last_selected_index = -1

is_j_mode = False
is_jumping = []

def mark_all_points():
    marked_image = original_image.copy()
    for index in range(len(selected_pixels)):
        pixel = selected_pixels[index]
        x = int(pixel[0])
        y = int(pixel[1])
        marked_image = imageMarker.mark_image_at_point(marked_image, y, x, 9, imageMarker.colors[index])
    return marked_image

def mouse_click(event, x, y, flags, param):
    global image, selected_pixels, editing_index, is_jumping, is_j_mode, last_selected_index
    if event == 1: #mouse click
        for index in range(len(selected_pixels)):
            pixel = selected_pixels[index]
            selected_x = pixel[0]
            selected_y = pixel[1]
            if selected_x - 4 <= x and x <= selected_x + 4 and selected_y - 4 <= y and y <= selected_y + 4:
                editing_index = index
                return

        if len(selected_pixels) < len(imageMarker.colors):
            selected_pixels.append([x, y])
            last_selected_index = len(selected_pixels) - 1
            image = mark_all_points()
    elif event == 4: #mouse up
        if editing_index > -1:
            last_selected_index = editing_index
            selected_pixels[editing_index] = [x, y]
            image = mark_all_points()
            # mark that the player is jumping at this frame
            if is_j_mode:
                is_jumping[editing_index] = True
                is_j_mode = False
            editing_index = -1


def handpick_image(img, estimated_pixels = []):
    global image, original_image, selected_pixels, is_jumping, is_j_mode, last_selected_index
    selected_pixels = estimated_pixels
    original_image = img
    is_jumping = [False for i in range(4)]
    image = img.copy()
    is_j_mode = False
    last_selected_index = -1
    _, width, _ = original_image.shape

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
            is_jumping = [False for i in range(4)]

        if key == ord('j'):
            is_j_mode = True

        if key == ord('c'):
            break

        if last_selected_index < len(selected_pixels) and last_selected_index != -1:
            if key == 82: #Up
                selected_pixels[last_selected_index][1] = max(0, selected_pixels[last_selected_index][1] - 1)
                image = mark_all_points()

            if key == 84: #Down
                selected_pixels[last_selected_index][1] = min(width, selected_pixels[last_selected_index][1] + 1)
                image = mark_all_points()

            if key == 81: #Left
                selected_pixels[last_selected_index][0] = max(0, selected_pixels[last_selected_index][0] - 1)
                image = mark_all_points()

            if key == 83: #Right
                selected_pixels[last_selected_index][0] = min(width, selected_pixels[last_selected_index][0] + 1)
                image = mark_all_points()

    return selected_pixels, is_jumping
