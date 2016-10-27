import cv2
import numpy as np

def convertGrayscaleImagesToVideo(images, fps, videoName):
    height , width =  images[0].shape

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    video = cv2.VideoWriter(videoName + '.avi',fourcc,fps,(width,height))

    for i in range(len(images)):
        rgb_image = cv2.cvtColor(np.uint8(images[i]), cv2.COLOR_GRAY2BGR)
        video.write(rgb_image)
    video.release()

def convertImagesToVideo(images, fps, videoName):
    height, width, _ = images[0].shape

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    video = cv2.VideoWriter(videoName + '.avi',fourcc,fps,(width,height))

    for i in range(len(images)):
        video.write(images[i])
    video.release()

# images = []
# for i in range(652):
#     images.append(cv2.imread('beachVolleyball1/edge/frame' + str(i) + '.jpg'))
#
# height , width, layer =  images[0].shape
#
# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# video = cv2.VideoWriter('beachVolleyball1/beachVolleyball1_edge.avi',fourcc,59,(width,height))
#
# for i in range(652):
#     video.write(images[i])
# video.release()
