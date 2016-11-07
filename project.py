import getopt
import numpy as np
import numpy.linalg as la
import os
import sys
import multiprocessing
from joblib import Parallel, delayed

import cv2
import cv2.cv as cv

import changeDetection
import cornerDetection
import edgeDetection
import handpickPixel
import imagesToVideo
import util


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

def homography_mapping(video_image, first_frame, homography_matrix):
    """
    Calculate the new position of where the pixel should be at after multiplying the original position with the
    homography matrix
    """
    new_video_image = np.zeros_like(first_frame)
    height, width, _ = first_frame.shape
    for h in range(height):
        for w in range(width):
            # convert pixel to 3-D point by appending depth as 1
            point = np.append([h, w], [1])
            new_pos_x, new_pos_y, new_pos_z = np.dot(homography_matrix, point)
            new_pos_x, new_pos_y = int(new_pos_x), int(new_pos_y)
            try:
                new_video_image[new_pos_x][new_pos_y] = video_image[h][w]
            except IndexError:
                continue
    return new_video_image

def getLastCoordinatesWithStatusArr(coordinates, status_arr):
    last_coordinates = []
    for coord_index in range(len(coordinates[-1])):
        for st_index in range(len(status_arr) - 1, -1, -1):
            if status_arr[st_index][coord_index] == 1:
                last_coordinates.append(coordinates[st_index][coord_index])
                break
        if len(last_coordinates) - 1 < coord_index:
            last_coordinates.append([50, 50])
    return last_coordinates

def inverse_homography_mapping(video_image, first_frame, inverse_homography_matrix):
    new_video_image = np.zeros_like(first_frame)
    height, width, _ = first_frame.shape
    for h in range(height):
        for w in range(width):
            # convert pixel to 3-D point by appending depth as 1
            point = np.append([h, w], [1])
            new_pos_x, new_pos_y, new_pos_z = np.dot(inverse_homography_matrix, point)
            new_pos_x, new_pos_y = int(new_pos_x), int(new_pos_y)
            try:
                new_video_image[h][w] = video_image[new_pos_x][new_pos_y]
            except IndexError:
                continue
    return new_video_image

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
        height, width, _ = first_frame.shape

        marked_images = []
        estimated_pixels = []
        all_selected_pixels = []
        skip_frame = 20
        for start_index in range(0, len(video_images), skip_frame):
            cv2.destroyAllWindows()
            if start_index > 0:
                cv2.imshow(str(start_index - skip_frame), marked_images[-skip_frame])
            start_frame = video_images[start_index]
            selected_pixels = handpickPixel.handpick_image(start_frame, estimated_pixels)
            all_selected_pixels.append(selected_pixels)
            temp_marked_images, marked_frame_coordinates, status_arr = changeDetection.mark_features_on_all_images(video_images[start_index: start_index + skip_frame + 1], selected_pixels)
            estimated_pixels = getLastCoordinatesWithStatusArr(marked_frame_coordinates, status_arr)
            marked_images = marked_images + temp_marked_images
        cv2.destroyAllWindows()

        util.save_coordinates(video_file_name, all_selected_pixels)
        # all_selected_pixels = util.load_coordinates(video_file_name)

        homography_matrixes = []
        # skip the first frame
        for selected_pixel in all_selected_pixels[1:]:
            # H = homography.find_homography(marked_frame_coordinates[0], mark_frame_coordinate)
            H, inliers = cv2.findHomography(np.float32(all_selected_pixels[0]), np.float32(selected_pixel), cv.CV_RANSAC)
            homography_matrixes.append(H)

        inverse_homography_matrixes = []
        # calculate the homography inverse
        for homography_matrix in homography_matrixes:
            inverse_homography_matrix = la.inv(homography_matrix)
            inverse_homography_matrixes.append(inverse_homography_matrix)

        new_video_images = []
        new_video_images.append(first_frame)

        # paralleling the homography mapping
        num_cores = multiprocessing.cpu_count()
        new_video_images = Parallel(n_jobs=num_cores, verbose=11)(delayed(homography_mapping)(video_images[(i+1) * skip_frame], first_frame, homography_matrixes[i]) for i in range(len(homography_matrixes)))
        # new_video_images = Parallel(n_jobs=num_cores, verbose=11)(delayed(inverse_homography_mapping)(video_images[(i+1) * skip_frame], first_frame, inverse_homography_matrixes[i], i) for i in range(len(inverse_homography_matrixes)))

        print 'Done with homography calculation. Writing to file now...'
        video_path = os.path.join(video_file_name, video_file_name + '_homography_orig')
        imagesToVideo.images_to_video(new_video_images, fps / skip_frame, video_path)

        # video_path = os.path.join(video_file_name, video_file_name + '_traced')
        # imagesToVideo.images_to_video(marked_images, fps, video_path)

    else:
        print 'Operation is not supported.'
        sys.exit(2)


if __name__ == '__main__':
    main()
