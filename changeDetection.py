import numpy as np
import cv2

lucas_kanade_params = dict(
    winSize= (10, 10),
    maxLevel= 0, #level of pyramids used
    criteria= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# color = np.random.randint(0,255,(100,3))
color = np.uint8([255, 128, 0]) #blue


def mark_features_on_all_images(images, features_coordinates):
    marked_images = []
    marked_frame_coordinates = []

    last_gs_img = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)

    p0 = []
    for coordinate in features_coordinates:
        p0.append([coordinate,])
    p0 = np.float32(p0)

    mask = np.zeros_like(images[0])
    for fr in range(1, len(images)):
        marked_coordinates = []
        frame = images[fr]
        gs_img = cv2.cvtColor(images[fr], cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(last_gs_img, gs_img, p0, None, **lucas_kanade_params)

        new_points = p1[st==1]
        old_points = p0[st==1]

        for point in new_points:
            x, y = point.ravel()
            marked_coordinates.append([x,y])
            cv2.circle(frame, (x, y), 5, color.tolist(), -1)
        marked_frame_coordinates.append(marked_coordinates)

        img = cv2.add(frame,mask)
        marked_images.append(img)

        # update last frame and point
        last_gs_img = gs_img.copy()
        p0 = new_points.reshape(-1,1,2)

    return marked_images, marked_frame_coordinates
