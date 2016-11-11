import matplotlib
matplotlib.use('TkAgg')

import math 
import numpy as np
import matplotlib.pyplot as plt
import util

def calculate_distance(pointA, pointB):
    # length of an actual beach volleyball court (in meters)
    standard_court_length = 16
    length_pixel = 718

    A_x, A_y = pointA[0], pointA[1]
    B_x, B_y = pointB[0], pointB[1]
    return math.sqrt((B_x - A_x) * (B_x - A_x) + (B_y - A_y) * (B_y - A_y)) / length_pixel * standard_court_length

def generate_statistics(all_selected_players_feet, all_is_jumping):
    """
        Generate statistics of the match
        Distance travelled by each players, Number of jumps of each players
    """
    # calculate the distance travelled by 4 different players
    distance_travelled = [[0] for _ in range(4)]
    for idx, selected_players_feet in enumerate(all_selected_players_feet[:-1]):
        for i in range(min(4, len(selected_players_feet))):
            next_frame_player_position = all_selected_players_feet[idx + 1][i]
            distance = distance_travelled[i][-1] + calculate_distance(selected_players_feet[i],
                                                                      next_frame_player_position)
            distance_travelled[i].append(distance)

        for i in range(len(selected_players_feet), 4):
            distance_travelled[i].append(0)

    # calculate the number of time each player jumps each frame
    num_jumps_of_each_player = [[0] for _ in range(4)]
    for idx, is_jumping in enumerate(all_is_jumping[:-1]):
        for i in range(min(4, len(is_jumping))):
            jump_cnt = num_jumps_of_each_player[i][-1]
            if all_is_jumping[idx][i] is False and all_is_jumping[idx + 1][i] is True:
                jump_cnt += 1
            num_jumps_of_each_player[i].append(jump_cnt)

    return distance_travelled, num_jumps_of_each_player

def draw_stats_table(distances, jumps, video_file_name):
    statsImages = []
    video_file_name = video_file_name + "_stats"

    _, N = np.shape(distances)

    for i in xrange(N):
        fig = plt.figure()
        plt.axis('off')
        ax = plt.gca()

        colLabels = ['Player', 'Distance(m)', 'Jump']
        rowLabels = ['', '', '', '']

        tableValues =[['',round(distances[0][i], 2),jumps[0][i]],
                        ['',round(distances[1][i], 2),jumps[1][i]],
                        ['',round(distances[2][i], 2),jumps[2][i]],
                        ['',round(distances[3][i], 2),jumps[3][i]]]

        # colours for each players in the following order: Red, Yellow, Green, Blue
        colours = [[(0, 0, 0.8), (1, 1, 1), (1, 1, 1)],
                    [(0, 1, 1), (1, 1, 1), (1, 1, 1)],
                    [(0, 0.8, 0), (1, 1, 1), (1, 1, 1)],
                    [(0.8, 0, 0), (1, 1, 1), (1, 1, 1)]]
       
        the_table = plt.table(cellText=tableValues, cellColours=colours, rowLoc='right', rowLabels=rowLabels,
                             colWidths=[.3]*3, colLoc='center', colLabels=colLabels, loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(20)
        the_table.scale(1, 6)
        fig.canvas.draw()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        statsImages.append(data)

        plt.clf()
        plt.close(fig)
        
    util.images_to_video(statsImages, 60, video_file_name) 