import pickle
import os
import imagesToVideo
import cv2
import cv2.cv as cv

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

def save_is_jumping(video_file, coordinates):
    video_path = os.path.join(video_file, video_file + '_is_jumping.pickle')
    with open(video_path, 'wb') as f:
        pickle.dump(coordinates, f, pickle.HIGHEST_PROTOCOL)

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
    print "cut_index", cut_index
    for fr in range(0, cut_index):
        _, img = cap.read()
        images.append(img)

    print "len of images: ", len(images)
    imagesToVideo.images_to_video(images, fps, first_part_filename)

    images = []
    for fr in range(cut_index, frame_count):
        _, img = cap.read()
        images.append(img)

    imagesToVideo.images_to_video(images, fps, second_part_filename)

    cap.release()
