import getopt
import numpy as np
import os
import sys

import cv2
import cv2.cv as cv

import changeDetection
import cornerDetection
import edgeDetection
import handpickPixel
import homography
import imagesToVideo


def video_to_sobel_edge_detection(video_file):
    video_images, fps = get_all_frame_images_and_fps(video_file)
    video_file_name, _ = video_file.split('.')
    if not os.path.isdir('./' + video_file_name + '/edge'):
        os.mkdir(video_file_name + '/edge')

    edge_images = edgeDetection.detect_edges(video_images, video_file_name)
    return edge_images, fps

def get_all_frame_images_and_fps(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CV_CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

    print('Frame Width ', frame_width)
    print('Frame Height ', frame_height)
    print('FPS ', fps)
    print('Frame Count', frame_count)

    images = []
    for fr in range(0, frame_count):
        _, img = cap.read()
        images.append(img)
    cap.release()
    return images, fps

def mark_corners_on_all_images(images, folder_name):
    marked_images = []
    marked_frame_coordinates = []
    for i in range(len(images)):
    # for i in range(3):
        print('frame', i, 'out of', len(images))
        image_name = os.path.join(folder_name, 'corners', 'frame' + str(i) + '.jpg')
        marked_image, marked_coordinates = cornerDetection.markCornerOnImage(images[i], image_name)
        marked_frame_coordinates.append(marked_coordinates)
        marked_images.append(marked_image)
    return marked_images

def video_to_corner_detection(video_file):
    video_images, fps = get_all_frame_images_and_fps(video_file)
    video_file_name, _ = video_file.split('.')
    if not os.path.isdir('./' + video_file_name + '/corners'):
        os.mkdir(video_file_name + '/corners')
    marked_images = mark_corners_on_all_images(video_images, video_file_name)
    return marked_images, fps

def initFolder(video_file):
    video_file_name, _ = video_file.split('.')
    if not os.path.isdir('./' + video_file_name):
        os.mkdir(video_file_name)

def usage():
    print "usage: " + sys.argv[0] + \
        " -o <operation>" + \
        " -f <filename>"

def main():
    video_file = operation = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'o:f:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-o':
            operation = a
        elif o == '-f':
            video_file = a
        else:
            assert False, "unhandled option"

    if video_file is None or operation is None:
        usage()
        sys.exit(2)

    video_file_name, _ = video_file.split('.')
    initFolder(video_file)

    if operation == 'edge':
        images, fps = video_to_sobel_edge_detection(video_file)
        video_path = os.path.join(video_file_name, video_file_name + '_sobel_edge')
        imagesToVideo.grayscale_image_to_video(images, fps, video_path)
    elif operation == 'corner':
        images, fps = video_to_corner_detection(video_file)
        video_path = os.path.join(video_file_name, video_file_name + '_corner')
        imagesToVideo.images_to_video(images, fps, video_path)
    elif operation == 'handpick':
        video_images, fps = get_all_frame_images_and_fps(video_file)
        first_frame = video_images[0]
        selected_pixels = handpickPixel.handpick_image(first_frame)
        selected_pixels = [[192, 274], [167, 243], [130, 196], [342, 107], [435, 136]]
        height, width, _ = first_frame.shape
        marked_images, marked_frame_coordinates = changeDetection.mark_features_on_all_images(video_images, selected_pixels)
        print(np.array(marked_frame_coordinates))

        homography_matrixes = []
        # skip the first frame
        for mark_frame_coordinate in marked_frame_coordinates[1:300]:
            # H = homography.find_homography(marked_frame_coordinates[0], mark_frame_coordinate)
            H, inliers = cv2.findHomography(np.float32(marked_frame_coordinates[0]), np.float32(mark_frame_coordinate), cv.CV_RANSAC)
            homography_matrixes.append(H)

        new_video_images = []
        new_video_images.append(first_frame)
        for idx, video_image in enumerate(video_images[1:300]):
            new_video_image = np.zeros_like(first_frame)
            for h in range(height):
                for w in range(width):
                    # convert pixel to 3-D point by appending depth as 1
                    point = np.append([h, w], [1])
                    new_pos_x, new_pos_y, new_pos_z = np.dot(homography_matrixes[idx], point)
                    new_pos_x, new_pos_y = int(new_pos_x), int(new_pos_y)
                    try:
                        new_video_image[new_pos_x][new_pos_y] = video_image[h][w]
                    except IndexError:
                        continue
            new_video_images.append(new_video_image)
            print 'Done with frame ', idx

        print 'Done with homography calculation. Writing to file now...'
        video_path = os.path.join(video_file_name, video_file_name + '_traced')
        imagesToVideo.images_to_video(new_video_images, fps, video_path)

    else:
        print 'Operation is not supported.'
        sys.exit(2)

if __name__ == '__main__':
    main()
