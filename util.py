import matplotlib
matplotlib.use('TkAgg')

import pickle
import os
import imagesToVideo
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import math

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

def drawStatsTable(distances, jumps, video_file_name):
    statsImages = []
    video_file_name = video_file_name + "_stats"

    _, N = np.shape(distances)
    for i in xrange(N):
        fig = plt.figure()
        plt.axis('off')
        ax = plt.gca()

        colLabels = ['Player', 'Travel Distance(m)', 'Number of Jump']
        rowLabels = ['', '', '', '']

        tableValues =[['',round(distances[0][i], 2),jumps[0][i]],
                        ['',round(distances[1][i], 2),jumps[1][i]],
                        ['',round(distances[2][i], 2),jumps[2][i]],
                        ['',round(distances[3][i], 2),jumps[3][i]]]
        # colours for each players in the following order: Red, Yellow, Green, Blue
        colours = [[(0.8, 0, 0), (1, 1, 1), (1, 1, 1)],
                    [(1, 1, 0), (1, 1, 1), (1, 1, 1)],
                    [(0, 0.8, 0), (1, 1, 1), (1, 1, 1)],
                    [(0, 0, 0.8), (1, 1, 1), (1, 1, 1)]]
       
        the_table = plt.table(cellText=tableValues, cellColours=colours, rowLoc='right', rowLabels=rowLabels,
                             colWidths=[.3]*3, colLoc='center', colLabels=colLabels, loc='center')
        the_table.scale(2, 8)
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        statsImages.append(data)

        plt.clf()
        plt.close(fig)
        
        # plt.show()
    imagesToVideo.images_to_video(statsImages, 60, video_file_name)

