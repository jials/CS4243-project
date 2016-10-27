import numpy as np
import cv2

lucas_kanade_params = dict(
    winSize= (10, 10),
    maxLevel= 0, #level of pyramids used
    criteria= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# color = np.random.randint(0,255,(100,3))
color = np.uint8([255, 128, 0]) #blue


def markFeaturesOnAllImages(images, features_coordinates):
    result = []

    last_gs_img = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)

    p0 = []
    for coordinate in features_coordinates:
        p0.append([coordinate,])
    p0 = np.float32(p0)

    mask = np.zeros_like(images[0])
    for fr in range(1, len(images)):
        frame = images[fr]
        gs_img = cv2.cvtColor(images[fr], cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(last_gs_img, gs_img, p0, None, **lucas_kanade_params)

        new_points = p1[st==1]
        old_points = p0[st==1]

        # for i,(new,old) in enumerate(zip(new_point,old_point)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #     cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

        for point in new_points:
            x, y = point.ravel()
            cv2.circle(frame, (x, y), 5, color.tolist(), -1)

        img = cv2.add(frame,mask)
        result.append(img)
        # cv2.imshow('frame',img)

        # update last frame and point
        last_gs_img = gs_img.copy()
        p0 = new_points.reshape(-1,1,2)

        # k = cv2.waitKey(0) & 0xff
        # if k == ord('q'):
        #     break

    # cv2.destroyAllWindows()
    return result
