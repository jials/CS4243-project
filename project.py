import cv2
import cv2.cv as cv
import numpy as np
import sys
import os
import imagesToVideo
import handpickPixel
import changeDetection
import edgeDetection
import cornerDetection

def videoToSobelEdgeDetection(video_file_name, extension):
    video_images, fps = getAllFrameImagesAndFps(video_file_name, extension)
    if not os.path.isdir('./' + video_file_name):
        os.mkdir(video_file_name)
    if not os.path.isdir('./' + video_file_name + '/edge'):
        os.mkdir(video_file_name + '/edge')

    edge_images = edgeDetection.detectEdgesFromImages(video_images, video_file_name)
    return edge_images, fps

def getAllFrameImagesAndFps(video_file_name, extension):
    cap = cv2.VideoCapture(video_file_name + extension)
    frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CV_CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

    print('Frame Width ', frame_width)
    print('Frame Height ', frame_height)
    print('FPS ', fps)
    print('Frame Count', frame_count)

    images = []
    for fr in range(0,frame_count):
        _,img = cap.read()
        images.append(img)
    cap.release()
    return images, fps;

def markCornersOnAllImages(images, folder_name):
    marked_images = [];
    marked_frame_coordinates = [];
    for i in range(len(images)):
    # for i in range(3):
        print('frame', i, 'out of', len(images))
        image_name = os.path.join(folder_name, 'corners', 'frame' + str(i) + '.jpg')
        marked_image, marked_coordinates = cornerDetection.markCornerOnImage(images[i], image_name)
        marked_frame_coordinates.append(marked_coordinates)
        marked_images.append(marked_image)
    return marked_images;

def videoToCornerDetection(video_file_name, extension):
    video_images, fps = getAllFrameImagesAndFps(video_file_name, extension)
    if not os.path.isdir('./' + video_file_name):
        os.mkdir(video_file_name)
    if not os.path.isdir('./' + video_file_name + '/corners'):
        os.mkdir(video_file_name + '/corners')
    marked_images = markCornersOnAllImages(video_images, video_file_name)
    return marked_images, fps

arguments = sys.argv
if len(arguments) == 4:
    task = arguments[1]
    video_file_name = arguments[2]
    extension = arguments[3]

    if task == 'edge':
        images, fps = videoToSobelEdgeDetection(video_file_name, extension)
        video_path = os.path.join(video_file_name, video_file_name + '_sobel_edge')
        imagesToVideo.convertGrayscaleImagesToVideo(images, fps, video_path)
    elif task == 'corner':
        images, fps = videoToCornerDetection(video_file_name, extension)
        video_path = os.path.join(video_file_name, video_file_name + '_corner')
        imagesToVideo.convertImagesToVideo(images, fps, video_path)
    elif task == 'handpick':
        video_images, fps = getAllFrameImagesAndFps(video_file_name, extension)
        first_frame = video_images[0]
        selected_pixels = handpickPixel.handpickImage(first_frame)
        height, width, _ = first_frame.shape
        marked_images, marked_frame_coordinates = changeDetection.markFeaturesOnAllImages(video_images, selected_pixels)
        video_path = os.path.join(video_file_name, video_file_name + '_traced')
        imagesToVideo.convertImagesToVideo(marked_images, fps, video_path)
        # print(np.array(marked_frame_coordinates))
