import pickle
import os

def load_coordinates(video_file):
    video_path = os.path.join(video_file, video_file + '.pickle')
    with open(video_path, 'rb') as f:
        coordinates = pickle.load(f)
    return coordinates

def save_coordinates(video_file, coordinates):
    video_path = os.path.join(video_file, video_file + '.pickle')
    with open(video_path, 'wb') as f:
        pickle.dump(coordinates, f, pickle.HIGHEST_PROTOCOL)
