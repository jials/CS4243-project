import getopt
import math
import numpy as np
import numpy.linalg as la
import os
import sys
import math

import cv2
import cv2.cv as cv

import changeDetection
import cornerDetection
import edgeDetection
import handpickPixel
import util
import imageMarker
import statistics

def video_to_sobel_edge_detection(video_file):
    video_images, fps = util.get_all_frame_images_and_fps(video_file)
    video_file_name, _ = video_file.split('.')
    if not os.path.isdir('./' + video_file_name + '/edge'):
        os.mkdir(video_file_name + '/edge')

    edge_images = edgeDetection.detect_edges(video_images, video_file_name)
    return edge_images, fps

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
    video_images, fps = util.get_all_frame_images_and_fps(video_file)
    video_file_name, _ = video_file.split('.')
    if not os.path.isdir('./' + video_file_name + '/corners'):
        os.mkdir(video_file_name + '/corners')
    marked_images = mark_corners_on_all_images(video_images, video_file_name)
    return marked_images, fps

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
        # tweak for vid 5
        # if index == len(other_images) - 1:
        #     continue
        translation_matrix = np.array([[1,0,-top_left[0]],[0,1,-top_left[1]],[0,0,1]])
        transformed_img = cv2.warpPerspective(image, np.dot(translation_matrix, H_arr[index]), transformed_shape)

        gray_transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY) # image to add in existing result
        gray_transformed_img = cv2.erode(gray_transformed_img, kernel, iterations = 1)
        _, mask = cv2.threshold(gray_transformed_img, 10, 255, cv2.THRESH_BINARY)
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

    return panaroma_image, stitch_results

def usage():
    print "usage: " + sys.argv[0] + \
        " -o <operation>" + \
        " -f <filename>" + \
        " [optional] -s <starting_frame_for_handpick_or_topdown> " + \
        " [optional] -d <destination_video_for_video_stitching> " + \
        " [optional] -t <second_for_video_cutting>"

def main():
    video_file = operation = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'o:f:s:t:d:')
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
        elif o == '-d':
            destination_video = int(a)
        else:
            assert False, "unhandled option"

    if video_file is None or operation is None:
        usage()
        sys.exit(2)

    if operation != 'concatenate' and operation != 'stitch':
        video_file_name, _ = video_file.split('.')
        util.initFolder(video_file)

    if operation == 'edge':
        images, fps = video_to_sobel_edge_detection(video_file)
        video_path = os.path.join(video_file_name, video_file_name + '_sobel_edge')
        util.grayscale_image_to_video(images, fps, video_path)

    elif operation == 'corner':
        images, fps = video_to_corner_detection(video_file)
        video_path = os.path.join(video_file_name, video_file_name + '_corner')
        util.images_to_video(images, fps, video_path)

    elif operation == 'handpick':
        if len(opts) < 3:
            starting_frame = 0

        video_images, fps = util.get_all_frame_images_and_fps(video_file)
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
                selected_pixels = handpickPixel.handpick_image(start_frame, estimated_pixels)[0]
                all_selected_pixels.append(selected_pixels)
                temp_marked_images, marked_frame_coordinates, status_arr = changeDetection.mark_features_on_all_images(video_images[start_index: min(len(video_images), start_index + skip_frame + 1)], selected_pixels)
                estimated_pixels = get_last_coordinates_with_status_arr(marked_frame_coordinates, status_arr)
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
            util.images_to_video(stitch_results, 3, video_path)

        if len(panaroma_image) > 0:
            image_name = os.path.join(video_file_name, video_file_name + '.jpg')
            cv2.imwrite(image_name, panaroma_image)

        # print 'Done with homography calculation. Writing to file now...'
        video_path = os.path.join(video_file_name, video_file_name + '_homography')
        util.images_to_video(new_video_images, fps / skip_frame, video_path)

        if len(marked_images) > 0:
            video_path = os.path.join(video_file_name, video_file_name + '_traced')
            util.images_to_video([marked_images[index] for index in range(0, len(marked_images), skip_frame)], fps/skip_frame, video_path)

    elif operation == 'topdown':
        if len(opts) < 3:
            starting_frame = 0

        court_image = cv2.imread('court.png')
        """
        work based on the assumption that points are picked by the following order:
        - top left, bottom left, top right, bottom right
        """
        # selected_court_pixels = handpickPixel.handpick_image(court_image)[0]
        selected_court_pixels = [[137, 140], [137, 498], [855, 139], [854, 498]]

        stitched_video_path = os.path.join(video_file_name, video_file_name + '.avi')
        video_images, fps = util.get_all_frame_images_and_fps(stitched_video_path)

        height, width, _ = court_image.shape

        # select court corners from the panorama video
        selected_court_corners = handpickPixel.handpick_image(video_images[0])[0]

        H, inliers = cv2.findHomography(np.float32(selected_court_corners), np.float32(selected_court_pixels), cv.CV_RANSAC)

        estimated_pixels = []
        all_selected_players_feet = []
        all_is_jumping = []
        ball_positions = []

        if starting_frame == -1:
            all_selected_players_feet = util.load_players_feet(video_file_name)
            all_is_jumping = util.load_is_jumping(video_file_name)
            ball_positions = util.load_ball_positions(video_file_name)
        else:
            for index in range(0, len(video_images)):
                cv2.destroyAllWindows()
                if index > 0:
                    cv2.imshow(str(index - 1), imageMarker.mark_image_at_points(video_images[index - 1], selected_players_feet, 9))
                selected_players_feet, is_jumping, player_index_with_ball = handpickPixel.handpick_image(video_images[index], estimated_pixels)

                all_is_jumping.append(is_jumping)
                if player_index_with_ball == 4 and len(selected_players_feet) == 5:
                    #the fifth point is the ball (use when the ball hits the ground)
                    ball_positions.append(selected_players_feet[-1])
                    selected_players_feet = selected_players_feet[:4]
                else:
                    ball_positions.append(selected_players_feet[player_index_with_ball] if player_index_with_ball != -1 else [-1, -1])
                all_selected_players_feet.append(selected_players_feet)

                #use change detection estimate next frame position
                if not index == len(video_images) - 1:
                    temp_marked_images, marked_frame_coordinates, status_arr = changeDetection.mark_features_on_all_images(
                        video_images[index: index + 2], selected_players_feet)
                    estimated_pixels = marked_frame_coordinates[-1]

            cv2.destroyAllWindows()

            util.save_players_feet(video_file_name, all_selected_players_feet)
            util.save_is_jumping(video_file_name, all_is_jumping)
            util.save_ball_positions(video_file_name, ball_positions)

        # colors for each players in the following order: Red, Yellow, Green, Blue
        colors = [(0, 0, 204), (0, 255, 255), (0, 204, 0), (204, 0, 0)]

        has_ball_positions = [False if ball_position[0] == -1 and ball_position[1] == -1 else True for ball_position in ball_positions]
        ball_positions = cv2.perspectiveTransform(np.float32(ball_positions).reshape(-1, 1, 2), H).reshape(-1, 2)
        for idx, selected_players_feet in enumerate(all_selected_players_feet):
            all_selected_players_feet[idx] = cv2.perspectiveTransform(np.float32(selected_players_feet).reshape(-1, 1, 2), H).reshape(-1, 2)

        #fill up frame without ball position
        indexes_with_ball_positions = np.where(np.array(has_ball_positions) == True)[0]
        # print('indexes with ball positions', indexes_with_ball_positions)
        for idx in range(0, indexes_with_ball_positions[0]):
            ball_positions[idx] = [0, 0]
        for idx in range(indexes_with_ball_positions[-1] + 1, len(has_ball_positions)):
            ball_positions[idx] = [0, 0]
        for start_index, end_index in zip(indexes_with_ball_positions[:-1], indexes_with_ball_positions[1:]):
            start_position = np.array(ball_positions[start_index])
            end_position = np.array(ball_positions[end_index])
            for idx in range(start_index + 1, end_index):
                ratio = float(idx - start_index) / (end_index - start_index)
                new_position = (1 - ratio) * start_position + ratio * end_position
                ball_positions[idx] = new_position
                has_ball_positions[idx] = True

        # add more frames to smoothen the videos
        frames_to_add = 19
        projected_all_selected_players_feet = [all_selected_players_feet[0]]
        projected_all_is_jumping = [all_is_jumping[0]]
        projected_ball_positions = [ball_positions[0]]
        projected_has_ball_positions = [has_ball_positions[0]]
        for idx in range(len(all_selected_players_feet) - 1):
            cur_players_feet = all_selected_players_feet[idx]
            next_players_feet = all_selected_players_feet[idx + 1]
            cur_is_jumping = all_is_jumping[idx]
            next_is_jumping = all_is_jumping[idx + 1]
            cur_ball_position = ball_positions[idx]
            next_ball_position = ball_positions[idx + 1]

            for fr in range(frames_to_add):
                ratio = float(fr) / frames_to_add

                #add frame info for feet
                new_players_feet = []
                for cur_player_feet, next_player_feet in zip(cur_players_feet, next_players_feet):
                    new_player_feet = (1 - ratio) * np.array(cur_player_feet) + ratio * np.array(next_player_feet)
                    new_players_feet.append(new_player_feet)
                projected_all_selected_players_feet.append(new_players_feet)

                #add frame info for jumping
                if ratio <= 0.5:
                    projected_all_is_jumping.append(cur_is_jumping)
                else:
                    projected_all_is_jumping.append(next_is_jumping)

                #add frame info for player with ball
                if has_ball_positions[idx] == True and has_ball_positions[idx + 1] == True:
                    projected_ball_positions.append((1 - ratio) * cur_ball_position + ratio * next_ball_position)
                    projected_has_ball_positions.append(True)
                else:
                    projected_ball_positions.append([-50, -50])
                    projected_has_ball_positions.append(False)

            projected_all_selected_players_feet.append(next_players_feet)
            projected_all_is_jumping.append(next_is_jumping)
            projected_ball_positions.append(next_ball_position)
            projected_has_ball_positions.append(has_ball_positions[idx + 1])

        all_selected_players_feet = projected_all_selected_players_feet
        all_is_jumping = projected_all_is_jumping
        ball_positions = projected_ball_positions
        has_ball_positions = projected_has_ball_positions

        # calculate the new position of the player with respect to top-down view
        court_images = []
        for selected_players_feet, is_jumping, ball_position, has_ball_position in zip(all_selected_players_feet, all_is_jumping, ball_positions, has_ball_positions):
            new_court_image = court_image.copy()
            #draw player position
            for idx, player_feet in enumerate(selected_players_feet):
                if idx >= 4:
                    break
                cv2.circle(new_court_image, tuple(player_feet), 10, colors[idx], 10)
                if is_jumping[idx]:
                    cv2.circle(new_court_image, tuple(player_feet), 10, (0, 0, 0), 6)

            #draw ball position
            if has_ball_position:
                cv2.circle(new_court_image, tuple(ball_position), 4, (255, 255, 255), 4)

            court_images.append(new_court_image)
        #     cv2.imshow('court image', new_court_image)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()
        video_path = os.path.join(video_file_name, video_file_name + '_court')
        util.images_to_video(court_images, fps * (frames_to_add + 1), video_path)
        # util.images_to_video(court_images, fps, video_path)

        distance_travelled, num_jumps_of_each_player = statistics.generate_statistics(all_selected_players_feet, all_is_jumping)

        statistics.draw_stats_table(distance_travelled, num_jumps_of_each_player, video_file_name)

    elif operation == "cut":
        if len(opts) < 3:
            print "Second of video to cut is not indicated."
            return

        util.cut_video(video_file, cut_second)

        print "Video is cut successfully."

    elif operation == 'stitch':
        if len(video_file.split(',')) != 2:
            print('Please give two file names for -f')
            exit(2)
        elif destination_video is None or destination_video < 1 or destination_video > 2:
            print('Invalid destination_video')
            exit(2)

        video1, video2 = video_file.split(',')
        video1_folder_name, video1_index = video1.split('.')[0].split('_')
        video2_folder_name, video2_index = video2.split('.')[0].split('_')
        video1_path = os.path.join(video1_folder_name, video1)
        video2_path = os.path.join(video2_folder_name, video2)
        video1_images, fps_1 = util.get_all_frame_images_and_fps(video1_path)
        video2_images, fps_2 = util.get_all_frame_images_and_fps(video2_path)

        cv2.imshow('second video first frame', video2_images[0])
        video1_selected_pixels = handpickPixel.handpick_image(video1_images[-1])[0]
        cv2.destroyAllWindows()
        cv2.imshow('previously picked position', imageMarker.mark_image_at_points(video1_images[-1], video1_selected_pixels, 9))
        video2_selected_pixels = handpickPixel.handpick_image(video2_images[0], [])[0]

        transformed_images = []
        if destination_video == 1:
            H, _ = cv2.findHomography(np.float32(video2_selected_pixels), np.float32(video1_selected_pixels), cv.CV_RANSAC)
            H_arr = []
            for _ in range(len(video1_images)):
                H_arr.append(np.identity(3))
            for _ in range(len(video2_images)):
                H_arr.append(H)
            panaroma_image, stitch_results = stitchImages(video1_images[-1], video1_images + video2_images, H_arr)
        else:
            H, _ = cv2.findHomography(np.float32(video1_selected_pixels), np.float32(video2_selected_pixels), cv.CV_RANSAC)
            H_arr = []
            for _ in range(len(video1_images)):
                H_arr.append(H)
            for _ in range(len(video2_images)):
                H_arr.append(np.identity(3))
            panaroma_image, stitch_results = stitchImages(video2_images[0], video1_images + video2_images, H_arr)

        cv2.imshow('panorama image', np.array(panaroma_image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        output_video_path = os.path.join(video1_folder_name, video1_folder_name + '_' + video1_index + 'n' + video2_index)
        util.images_to_video(stitch_results, int((fps_1 + fps_2) / 2), output_video_path)

    elif operation == "concatenate":
        video_files = video_file.split(",")
        video_file_name, _ = video_files[0].split('.')
        util.concatenate_video(video_files, video_file_name)

    else:
        print 'Operation is not supported.'
        sys.exit(2)

if __name__ == '__main__':
    main()
