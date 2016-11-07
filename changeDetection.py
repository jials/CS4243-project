import numpy as np
import cv2
import imageMarker

lucas_kanade_params = dict(
    winSize= (4, 4),
    maxLevel= 0, #level of pyramids used
    criteria= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

red = np.uint8([0, 0, 255])
orange = np.uint8([0, 127, 255])
yellow = np.uint8([0, 255, 255])
green = np.uint8([0, 255, 0])
blue = np.uint8([255, 0, 0])
colors = [red, orange, yellow, green, blue]

def mark_features_on_all_images(images, features_coordinates):
    marked_images = []
    marked_frame_coordinates = []

    last_gs_img = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)

    p0 = []
    for coordinate in features_coordinates:
        p0.append([coordinate,])
    p0 = np.float32(p0)

    mask = np.zeros_like(images[0])
    status_arr = []
    for fr in range(1, len(images)):
        marked_coordinates = []
        frame = images[fr].copy()
        gs_img = cv2.cvtColor(images[fr], cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(last_gs_img, gs_img, p0, None, **lucas_kanade_params)

        status_arr.append(st)

        if p1 is None:
            continue

        new_points = []
        for index in range(len(p1)):
            if st[index] == 1:
                new_points.append(p1[index])
            else:
                new_points.append(p0[index])
        new_points = np.array(new_points)

        for index, point in enumerate(new_points):
            x, y = point.ravel()
            marked_coordinates.append([x,y])
            imageMarker.mark_image_at_point(frame, int(y), int(x), 9, colors[index])
        marked_frame_coordinates.append(marked_coordinates)

        img = cv2.add(frame,mask)
        marked_images.append(img)

        # update last frame and point
        last_gs_img = gs_img.copy()
        p0 = new_points.reshape(-1,1,2)

    return marked_images, marked_frame_coordinates, status_arr
