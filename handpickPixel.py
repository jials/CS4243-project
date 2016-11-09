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

is_b_mode = False
player_index_with_ball = -1

def mark_all_points():
    marked_image = original_image.copy()
    for index in range(len(selected_pixels)):
        pixel = selected_pixels[index]
        x = int(pixel[0])
        y = int(pixel[1])
        marked_image = imageMarker.mark_image_at_point(marked_image, y, x, 9, imageMarker.colors[index])
        #mark jump
        if index < len(is_jumping) and is_jumping[index]:
            cv2.circle(marked_image, (x, y), 3, (0, 0, 0), 3)

        #mark ball
        if player_index_with_ball == index:
            cv2.circle(marked_image, (x, y), 2, (200, 100, 0), 2)
    return marked_image

def mouse_click(event, x, y, flags, param):
    global image, selected_pixels, editing_index, is_jumping, is_j_mode
    global last_selected_index, is_b_mode, player_index_with_ball
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

            # mark that the player is jumping at this frame
            if is_j_mode:
                is_jumping[editing_index] = True
                is_j_mode = False
            elif is_b_mode:
                player_index_with_ball = editing_index
                is_b_mode = False

            image = mark_all_points()
            editing_index = -1

def initiate_global_value():
    global image, original_image
    global is_jumping, is_j_mode, last_selected_index
    global is_b_mode, player_index_with_ball
    image = original_image.copy()
    is_jumping = [False for i in range(4)]
    is_j_mode = False
    is_b_mode = False
    last_selected_index = -1
    player_index_with_ball = -1

def handpick_image(img, estimated_pixels = []):
    global image, original_image, selected_pixels
    global is_jumping, is_j_mode, last_selected_index
    global is_b_mode, player_index_with_ball
    selected_pixels = estimated_pixels
    original_image = img
    initiate_global_value()

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
            initiate_global_value()

        if key == ord('j'):
            is_j_mode = True
            is_b_mode = False

        if key == ord('b'):
            is_b_mode = True
            is_j_mode = False

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

    return selected_pixels, is_jumping, player_index_with_ball
