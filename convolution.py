import numpy as np

def convolve(img, ff):
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
    return convolve(img, horizontal_sobel_filter)

def getSobelVerticalEdgeStrength(img):
    vertical_sobel_filter = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    return convolve(img, vertical_sobel_filter)
