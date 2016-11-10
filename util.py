import pickle
import os
import imagesToVideo
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt 

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

def concatenate_video(video_files):
    if not os.path.isdir('./concatenatedVideos'):
        os.mkdir('concatenatedVideos')

    videos = []

    for video in video_files:
        images, _ = get_all_frame_images_and_fps(video)
        videos.append(images)

    cap = cv2.VideoCapture(video_files[0])
    frame_width = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CV_CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

    print frame_width, "-", frame_height, "-", fps, "-", frame_count
    print "start concatenating videos"

    concatenatedVideo = []
    frameIncrement = 0
    currFrame = 0

    print len(videos[0])
    print len(videos[1])
    for i in xrange(frame_count):
        eachFrame = np.zeros((2 * frame_height, 2 * frame_width, 3), dtype="uint8")
        eachFrame[0:frame_height, 0:frame_width] = videos[0][i]
        if(frameIncrement == fps):
            currFrame += 1
            frameIncrement = 0
        eachFrame[0:frame_height, frame_width: frame_width+frame_width] = videos[1][currFrame]
        frameIncrement += 1
        concatenatedVideo.append(eachFrame)

    print "successfully concatenated videos"

    imagesToVideo.images_to_video(concatenatedVideo, fps, "concatenate_test")

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

def drawTable(data):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title('Statistic')

    colLabels = ['Travel Distance', 'Number of Jump']
    rowLabels = ['Player1', 'Player2', 'Player3', 'Player4']
    tableValues =[[11,12,13],[21,22,23],[31,32,33]]

    the_table = plt.table(cellText=tableValues, rowLoc='right', rowLabels=rowLabels,
                         colWidths=[.5,.5], colLabels=colLabels,
                         colLoc='center', loc='center')

    plt.show()
    # the_table = plt.table(cellText=tableValues, rowLoc='right',
                         # rowColours=colors, rowLabels=rowLabels,
                         # colWidths=[.5,.5], colLabels=colLabels,
                         # colLoc='center', loc='center')

