import pickle
import os
import cv2
import cv2.cv as cv
import numpy as np
import math

def initFolder(video_file):
    video_file_name, _ = video_file.split('.')
    if not os.path.isdir('./' + video_file_name):
        os.mkdir(video_file_name)

def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def load_coordinates(video_file):
    video_path = os.path.join(video_file, video_file + '.pickle')
    with open(video_path, 'rb') as f:
        coordinates = pickle.load(f)
    return coordinates

def save_coordinates(video_file, coordinates):
    video_path = os.path.join(video_file, video_file + '.pickle')
    with open(video_path, 'wb') as f:
        pickle.dump(coordinates, f, pickle.HIGHEST_PROTOCOL)

def load_players_feet(video_file):
    video_path = os.path.join(video_file, video_file + '_feet.pickle')
    with open(video_path, 'rb') as f:
        coordinates = pickle.load(f)
    return coordinates

def save_players_feet(video_file, coordinates):
    video_path = os.path.join(video_file, video_file + '_feet.pickle')
    with open(video_path, 'wb') as f:
        pickle.dump(coordinates, f, pickle.HIGHEST_PROTOCOL)

def load_is_jumping(video_file):
    video_path = os.path.join(video_file, video_file + '_is_jumping.pickle')
    with open(video_path, 'rb') as f:
        is_jumping = pickle.load(f)
    return is_jumping

def load_ball_positions(video_file):
    video_path = os.path.join(video_file, video_file + '_ball.pickle')
    with open(video_path, 'rb') as f:
        ball_positions = pickle.load(f)
    return ball_positions

def save_is_jumping(video_file, coordinates):
    video_path = os.path.join(video_file, video_file + '_is_jumping.pickle')
    with open(video_path, 'wb') as f:
        pickle.dump(coordinates, f, pickle.HIGHEST_PROTOCOL)

def save_ball_positions(video_file, ball_positions):
    video_path = os.path.join(video_file, video_file + '_ball.pickle')
    with open(video_path, 'wb') as f:
        pickle.dump(ball_positions, f, pickle.HIGHEST_PROTOCOL)

def concatenate_video(video_files, video_file_name):
    video_file_name = video_file_name + "_concatenated"
    cap = cv2.VideoCapture(video_files[0])
    frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CV_CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

    videos = []

    for idx, video in enumerate(video_files):
        images, _ = get_all_frame_images_and_fps(video)
        if idx > 0:
            resized_images = []
            for image in images:
                resized_image = cv2.resize(image, (frame_width, frame_height))
                resized_images.append(resized_image)

            videos.append(resized_images)
        else:
            videos.append(images)

    print "start concatenating videos"

    concatenatedVideo = []

    threshold1 = len(videos[1]) * 1.0 / len(videos[0])
    threshold2 = len(videos[2]) * 1.0 / len(videos[0])
    threshold3 = len(videos[3]) * 1.0 / len(videos[0])

    for i in xrange(frame_count):
        eachFrame = np.zeros((2 * frame_height, 2 * frame_width, 3), dtype="uint8")

        eachFrame[0:frame_height, 0:frame_width] = videos[0][i]

        frameIndex1 = int(round(threshold1 * i))
        frameIndex1 = frameIndex1 if frameIndex1 < len(videos[1]) else len(videos[1]) - 1

        eachFrame[0:frame_height, frame_width:frame_width+frame_width] = videos[1][frameIndex1]

        frameIndex2 = int(round(threshold2 * i))
        frameIndex2 = frameIndex2 if frameIndex2 < len(videos[2]) else len(videos[2]) - 1
        eachFrame[frame_height:frame_height+frame_height, 0:frame_width] = videos[2][frameIndex2]

        frameIndex3 = int(round(threshold3 * i))
        frameIndex3 = frameIndex3 if frameIndex3 < len(videos[3]) else len(videos[3]) - 1
        eachFrame[frame_height:frame_height+frame_height, frame_width:frame_width+frame_width] = videos[3][frameIndex3]

        concatenatedVideo.append(eachFrame)

    images_to_video(concatenatedVideo, fps, video_file_name)

    print "successfully concatenated videos"

    cap.release()

def cut_video(video_file, cut_second):
    video_file_name, _ = video_file.split('.')
    first_part_filename = video_file_name + "_first"
    second_part_filename = video_file_name + "_second"
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
    cut_index = int(cut_second  * fps)
    for fr in range(0, cut_index):
        _, img = cap.read()
        images.append(img)

    images_to_video(images, fps, first_part_filename)

    images = []
    for fr in range(cut_index, frame_count):
        _, img = cap.read()
        images.append(img)

    images_to_video(images, fps, second_part_filename)

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

def grayscale_image_to_video(images, fps, videoName):
    height , width =  images[0].shape

    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    video = cv2.VideoWriter(videoName + '.avi',fourcc,fps,(width,height))

    for i in range(len(images)):
        rgb_image = cv2.cvtColor(np.uint8(images[i]), cv2.COLOR_GRAY2BGR)
        video.write(rgb_image)
    video.release()

def images_to_video(images, fps, videoName):
    height, width, _ = images[0].shape

    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    video = cv2.VideoWriter(videoName + '.avi',fourcc,fps,(width,height))

    for i in range(len(images)):
        video.write(images[i])
    video.release()

def get_last_coordinates_with_status_arr(coordinates, status_arr):
    last_coordinates = []
    for coord_index in range(len(coordinates[-1])):
        for st_index in range(len(status_arr) - 1, -1, -1):
            if status_arr[st_index][coord_index] == 1:
                last_coordinates.append(coordinates[st_index][coord_index])
                break
        if len(last_coordinates) - 1 < coord_index:
            last_coordinates.append([50, 50])
    return last_coordinates
    
# unused homography mapping
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
            point = np.append([w, h], [1])
            new_pos_x, new_pos_y, new_pos_z = np.dot(homography_matrix, point)
            new_pos_x, new_pos_y = int(new_pos_x), int(new_pos_y)
            try:
                new_video_image[new_pos_y][new_pos_x] = video_image[h][w]
            except IndexError:
                continue
    return new_video_image

# unused inverse homography mapping
def inverse_homography_mapping(video_image, first_frame, inverse_homography_matrix):
    new_video_image = np.zeros_like(first_frame)
    height, width, _ = first_frame.shape
    for h in range(height):
        for w in range(width):
            # convert pixel to 3-D point by appending depth as 1
            point = np.append([w, h], [1])
            new_pos_x, new_pos_y, new_pos_z = np.dot(inverse_homography_matrix, point)
            new_pos_x, new_pos_y = int(new_pos_x), int(new_pos_y)
            try:
                new_video_image[h][w] = video_image[new_pos_y][new_pos_x]
            except IndexError:
                continue
    return new_video_image
