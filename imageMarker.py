import numpy as np

red = np.uint8([0, 0, 255])
orange = np.uint8([0, 127, 255])
yellow = np.uint8([0, 255, 255])
green = np.uint8([0, 255, 0])
blue = np.uint8([255, 0, 0])
indigo = np.uint8([130, 0, 75])
violet = np.uint8([255, 0, 139])
colors = [red, orange, yellow, green, blue, indigo, violet]

default_color = np.uint8([255, 128, 0])

def mark_image_at_point(img, y, x, size, color = default_color):
    if size < 1:
        size = 1
    negative_offset = (size - 1) / 2
    positive_offset = size / 2 + 1

    min_y = max(0, y - negative_offset)
    max_y = min(img.shape[0], y + positive_offset)
    min_x = max(0, x - negative_offset)
    max_x = min(img.shape[1], x + positive_offset)
    for y_offset in range(min_y, max_y):
        img[y_offset][min_x] = color
        img[y_offset][min_x + 1] = color
        img[y_offset][max_x - 1] = color
        img[y_offset][max_x - 2] = color

    for x_offset in range(min_x, max_x):
        img[min_y][x_offset] = color
        img[min_y + 1][x_offset] = color
        img[max_y - 1][x_offset] = color
        img[max_y - 2][x_offset] = color

    return img
 
def mark_image_at_points(img, coordinates, size):
    result = img.copy()
    for index, coordinate in enumerate(coordinates):
        result = mark_image_at_point(result, int(coordinate[1]), int(coordinate[0]), size, colors[index])
    return result



