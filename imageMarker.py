import numpy as np

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
