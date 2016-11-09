import getopt
import math
import numpy as np
import numpy.linalg as la
import os
import sys
import multiprocessing
from joblib import Parallel, delayed
import math

import cv2
import cv2.cv as cv

import changeDetection
import cornerDetection
import edgeDetection
import handpickPixel
import imagesToVideo
import util
import imageMarker


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
            point = np.append([w, h], [1])
            new_pos_x, new_pos_y, new_pos_z = np.dot(homography_matrix, point)
            new_pos_x, new_pos_y = int(new_pos_x), int(new_pos_y)
            try:
                new_video_image[new_pos_y][new_pos_x] = video_image[h][w]
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
            point = np.append([w, h], [1])
            new_pos_x, new_pos_y, new_pos_z = np.dot(inverse_homography_matrix, point)
            new_pos_x, new_pos_y = int(new_pos_x), int(new_pos_y)
            try:
                new_video_image[h][w] = video_image[new_pos_y][new_pos_x]
            except IndexError:
                continue
    return new_video_image

def stitchImages(base, other_images, H_arr):
    h1, w1, _ = base.shape
    base_corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)

    transformed_images = []
    min_x = 0
    min_y = 0
    max_x = w1
    max_y = h1

    for index, image in enumerate(other_images):
        h, w, _ = image.shape
        image_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        image_corners_after_h = cv2.perspectiveTransform(image_corners, H_arr[index])

        combined_corners = np.concatenate((base_corners, image_corners_after_h), axis = 0)
        temp_min_x = np.amin([corner[0][0] for corner in combined_corners])
        temp_min_y = np.amin([corner[0][1] for corner in combined_corners])
        temp_max_x = np.amax([corner[0][0] for corner in combined_corners])
        temp_max_y = np.amax([corner[0][1] for corner in combined_corners])
        # print(index)
        # print(temp_min_x)
        # print(temp_min_y)
        # print(temp_max_x)
        # print(temp_max_y)
        min_x = temp_min_x if temp_min_x < min_x else min_x
        min_y = temp_min_y if temp_min_y < min_y else min_y
        max_x = temp_max_x if temp_max_x > max_x else max_x
        max_y = temp_max_y if temp_max_y > max_y else max_y

    transformed_image_width = int(max_x - min_x)
    transformed_image_height = int(max_y - min_y)
    transformed_shape = (transformed_image_width, transformed_image_height)
    top_left = [min_x, min_y]
    print('panaroma image size', transformed_shape)
    if transformed_shape[0] > 5000 or transformed_shape[1] > 5000:
        print('image too big and skipped')
        return [], []

    result = np.zeros([transformed_shape[1], transformed_shape[0], 3])
    stitch_results = []
    kernel = np.ones((4,4),np.uint8)
    for index, image in enumerate(other_images):
        if index == len(other_images) - 1:
            continue
        translation_matrix = np.array([[1,0,-top_left[0]],[0,1,-top_left[1]],[0,0,1]])
        transformed_img = cv2.warpPerspective(image, np.dot(translation_matrix, H_arr[index]), transformed_shape)

        gray_transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY) # image to add in existing result
        gray_transformed_img = cv2.erode(gray_transformed_img, kernel, iterations = 1)
        _, mask = cv2.threshold(gray_transformed_img, 30, 255, cv2.THRESH_BINARY)
        mask_inverse = cv2.bitwise_not(mask)
        region_of_interest = cv2.bitwise_and(transformed_img, transformed_img, mask=mask)
        prev_result = cv2.bitwise_and(result, result, mask=mask_inverse)

        result = cv2.add(np.uint8(prev_result), np.uint8(region_of_interest))
        stitch_results.append(result)

    panaroma_image = result.copy()
    for index, stitched_image in enumerate(stitch_results):
        gray_stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY) # image to add in existing result
        _, mask = cv2.threshold(gray_stitched_image, 10, 255, cv2.THRESH_BINARY)
        mask_inverse = cv2.bitwise_not(mask)
        region_of_interest = cv2.bitwise_and(stitched_image, stitched_image, mask=mask)
        prev_result = cv2.bitwise_and(panaroma_image, panaroma_image, mask=mask_inverse)

        result = cv2.add(np.uint8(prev_result), np.uint8(region_of_interest))
        stitch_results[index] = result
    #     cv2.imshow(str(index), result)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return panaroma_image, stitch_results

def calculate_distance(pointA, pointB):
    # length of an actual beach volleyball court (in meters)
    standard_court_length = 16
    length_pixel = 324

    A_x, A_y = pointA[0], pointA[1]
    B_x, B_y = pointB[0], pointB[1]
    return math.sqrt((B_x - A_x) * (B_x - A_x) + (B_y - A_y) * (B_y - A_y)) / length_pixel * standard_court_length

def initFolder(video_file):
    video_file_name, _ = video_file.split('.')
    if not os.path.isdir('./' + video_file_name):
        os.mkdir(video_file_name)

def usage():
    print "usage: " + sys.argv[0] + \
        " -o <operation>" + \
        " -f <filename>" + \
        " [optional] -s <starting_frame_for_handpick> -t <second_for_video_cutting"

def main():
    video_file = operation = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'o:f:s:t:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-o':
            operation = a
        elif o == '-f':
            video_file = a
        elif o == '-s':
            starting_frame = int(a)
        elif o == '-t':
            cut_second = util.int_or_float(a)
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
        if len(opts) < 3:
            starting_frame = 0

        video_images, fps = get_all_frame_images_and_fps(video_file)
        first_frame = video_images[0]
        height, width, _ = first_frame.shape

        marked_images = []
        all_selected_pixels = util.load_coordinates(video_file_name) if starting_frame > 0 else []
        skip_frame = 20

        # remove those unwanted selected_pixels
        if starting_frame == -1:
            all_selected_pixels = util.load_coordinates(video_file_name)
        else:
            estimated_pixels = all_selected_pixels[int(starting_frame / skip_frame)] if starting_frame > 0 else []
            all_selected_pixels = all_selected_pixels[0: int(starting_frame / skip_frame)] if starting_frame > 0 else []

            jump_to_index = starting_frame - starting_frame % skip_frame

            #generate missing marked images
            for start_index in range(0, jump_to_index, skip_frame):
                selected_pixels = all_selected_pixels[start_index/skip_frame]
                temp_marked_images, marked_frame_coordinates, status_arr = changeDetection.mark_features_on_all_images(video_images[start_index: min(len(video_images), start_index + skip_frame + 1)], selected_pixels)
                marked_images = marked_images + temp_marked_images

            for start_index in range(jump_to_index, len(video_images), skip_frame):
                cv2.destroyAllWindows()
                if start_index > jump_to_index:
                    cv2.imshow(str(start_index - skip_frame), imageMarker.mark_image_at_points(video_images[start_index - skip_frame], all_selected_pixels[-1], 9))

                start_frame = video_images[start_index]
                selected_pixels, _ = handpickPixel.handpick_image(start_frame, estimated_pixels)
                all_selected_pixels.append(selected_pixels)
                temp_marked_images, marked_frame_coordinates, status_arr = changeDetection.mark_features_on_all_images(video_images[start_index: min(len(video_images), start_index + skip_frame + 1)], selected_pixels)
                estimated_pixels = getLastCoordinatesWithStatusArr(marked_frame_coordinates, status_arr)
                marked_images = marked_images + temp_marked_images
            cv2.destroyAllWindows()
            util.save_coordinates(video_file_name, all_selected_pixels)

        homography_matrixes = []
        # skip the first frame
        for selected_pixel in all_selected_pixels[1:]:
            # H = homography.find_homography(marked_frame_coordinates[0], mark_frame_coordinate)
            if len(selected_pixel) > len(all_selected_pixels[0]):
                print('warning: selected_pixel size is different with the all_selected_pixels[0] size')
                selected_pixel = selected_pixel[0: len(all_selected_pixels[0])]

            H, inliers = cv2.findHomography(np.float32(selected_pixel), np.float32(all_selected_pixels[0]), cv.CV_RANSAC)
            homography_matrixes.append(H)

        new_video_images = []
        new_video_images.append(first_frame)
        for index, matrix in enumerate(homography_matrixes):
            result = cv2.warpPerspective(video_images[(index + 1) * skip_frame], matrix, (width, height))
            new_video_images.append(result)

        skipped_images = [video_images[index] for index in range(skip_frame, len(video_images), skip_frame)]
        panaroma_image, stitch_results = stitchImages(first_frame, skipped_images, homography_matrixes)

        if len(stitch_results) > 0:
            video_path = os.path.join(video_file_name, video_file_name)
            imagesToVideo.images_to_video(stitch_results, 3, video_path)

        if len(panaroma_image) > 0:
            image_name = os.path.join(video_file_name, video_file_name + '.jpg')
            cv2.imwrite(image_name, panaroma_image)

        # print 'Done with homography calculation. Writing to file now...'
        video_path = os.path.join(video_file_name, video_file_name + '_homography')
        imagesToVideo.images_to_video(new_video_images, fps / skip_frame, video_path)

        if len(marked_images) > 0:
            video_path = os.path.join(video_file_name, video_file_name + '_traced')
            imagesToVideo.images_to_video([marked_images[index] for index in range(0, len(marked_images), skip_frame)], fps/skip_frame, video_path)

    elif operation == 'topdown':
        court_image = cv2.imread('court.png')
        """
        work based on the assumption that points are picked by the following order:
        - top left, bottom left, top right, bottom right
        """
        # selected_court_pixels, _ = handpickPixel.handpick_image(court_image)
        selected_court_pixels = [[63, 86], [60, 248], [388, 84], [386, 246]]

        stitched_video_path = os.path.join(video_file_name, video_file_name + '.avi')
        video_images, fps = get_all_frame_images_and_fps(stitched_video_path)

        height, width, _ = court_image.shape

        # select court corners from the panorama video
        selected_court_corners, _ = handpickPixel.handpick_image(video_images[0])

        H, inliers = cv2.findHomography(np.float32(selected_court_corners), np.float32(selected_court_pixels), cv.CV_RANSAC)

        estimated_pixels = []
        all_selected_players_feet = []
        all_is_jumping = []

        for index in range(0, len(video_images)):
            cv2.destroyAllWindows()
            if index > 0:
                cv2.imshow(str(index - 1), imageMarker.mark_image_at_points(video_images[index - 1], selected_players_feet, 9))
            selected_players_feet, is_jumping = handpickPixel.handpick_image(video_images[index], estimated_pixels)
            if not index == len(video_images) - 1:
                temp_marked_images, marked_frame_coordinates, status_arr = changeDetection.mark_features_on_all_images(
                    video_images[index: index + 2], selected_players_feet)
                estimated_pixels = marked_frame_coordinates[-1]
            all_selected_players_feet.append(selected_players_feet)
            all_is_jumping.append(is_jumping)
        cv2.destroyAllWindows()

        util.save_players_feet(video_file_name, all_selected_players_feet)
        util.save_is_jumping(video_file_name, all_is_jumping)
        # all_selected_players_feet = util.load_players_feet(video_file_name)
        # all_is_jumping = util.load_is_jumping(video_file_name)

        # colors for each players in the following order: Red, Yellow, Green, Blue
        colors = [(0, 0, 204), (0, 255, 255), (0, 204, 0), (204, 0, 0)]

        # add more frames to smoothen the videos
        frames_to_add = 19
        projected_all_selected_players_feet = [all_selected_players_feet[0]]
        projected_all_is_jumping = [all_is_jumping[0]]
        for idx in range(len(all_selected_players_feet) - 1):
            cur_players_feet = all_selected_players_feet[idx]
            next_players_feet = all_selected_players_feet[idx + 1]
            cur_is_jumping = all_is_jumping[idx]
            next_is_jumping = all_is_jumping[idx]

            for fr in range(frames_to_add):
                ratio = float(fr) / frames_to_add
                new_players_feet = []
                for cur_player_feet, next_player_feet in zip(cur_players_feet, next_players_feet):
                    new_player_feet = ratio * np.array(cur_player_feet) + (1 - ratio) * np.array(next_player_feet)
                    new_players_feet.append(new_player_feet)
                if ratio <= 0.5:
                    projected_all_is_jumping.append(cur_is_jumping)
                else:
                    projected_all_is_jumping.append(next_is_jumping)

                projected_all_selected_players_feet.append(new_players_feet)

            projected_all_selected_players_feet.append(next_players_feet)
            projected_all_is_jumping.append(next_is_jumping)

        all_selected_players_feet = projected_all_selected_players_feet
        all_is_jumping = projected_all_is_jumping
        # calculate the new position of the player with respect to top-down view
        court_images = []
        for selected_players_feet, is_jumping in zip(all_selected_players_feet, all_is_jumping):
            new_court_image = court_image.copy()
            for idx, player_feet in enumerate(selected_players_feet):
                if idx >= 4:
                    break
                point = np.transpose(np.matrix(np.append(player_feet, [1])))
                x, y, z = H * point
                x = x/z
                y = y/z
                cv2.circle(new_court_image, (x, y), 5, colors[idx], 6)
                if is_jumping[idx]:
                    cv2.circle(new_court_image, (x, y), 5, (0, 0, 0), 3)
            court_images.append(new_court_image)
            cv2.imshow('court image', new_court_image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        video_path = os.path.join(video_file_name, video_file_name + '_court')
        imagesToVideo.images_to_video(court_images, fps * (frames_to_add + 1), video_path)

        # calculate the distance travelled by 4 different players
        distance_travelled = [[0] for _ in range(4)]
        for idx, selected_players_feet in enumerate(all_selected_players_feet[:-1]):
            for i in range(len(selected_players_feet)):
                next_frame_player_position = all_selected_players_feet[idx+1][i]
                distance = distance_travelled[i][-1] + calculate_distance(selected_players_feet[i], next_frame_player_position)
                distance_travelled[i].append(distance)

            for i in range(len(selected_players_feet), 4):
                distance_travelled[i].append(0)

    elif operation == "cut":
        if len(opts) < 3:
            print "Second of video to cut is not indicated."
            return

        util.cut_video(video_file, cut_second)

        print "Video is cut successfully."
    else:
        print 'Operation is not supported.'
        sys.exit(2)


if __name__ == '__main__':
    main()
