import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
import imageio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


def get_num_frames(filename):
    cmd = f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format csv {filename}"
    num_frames = os.popen(cmd).read()
    num_frames = int(num_frames.split(',')[1])
    return num_frames


def get_frame(filename, frame_num):
    reader = imageio.get_reader(filename, "ffmpeg")
    reader.set_image_index(frame_num)
    frame = np.array(reader.get_next_data())
    reader.close()
    return frame


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, person_id=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.person_id = person_id

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if person_id is None:
                if id_sampling:
                    train_videos = {os.path.basename(video).split('#')[0] for video in
                                    os.listdir(os.path.join(root_dir, 'train'))}
                    train_videos = list(train_videos)
                else:
                    train_videos = os.listdir(os.path.join(root_dir, 'train'))
                print("number of train videos", len(train_videos))
                test_videos = os.listdir(os.path.join(root_dir, 'test'))
            else:
                print("Training on specific person", person_id)
                train_videos = ["id" + str(person_id)]
                test_videos = ["id" + str(person_id)]
            
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if (self.is_train and self.id_sampling) or (self.person_id is not None):
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            num_frames = get_num_frames(path)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
            num_frames)
            video_array = np.array([get_frame(path, frame_idx[0]), get_frame(path, frame_idx[1])]) 

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = video_array
            out['video'] = video
            out['video_path'] = str(path)
        
        out['name'] = video_name

        return out

class MetricsDataset(Dataset):
    """
        Load a select set of frames for computing consistent metrics/visuals on
    """

    def __init__(self, root_dir, frame_shape):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        file_name = self.videos[idx]
        path = os.path.join(self.root_dir, file_name)
        assert os.path.isdir(path)

        driving = img_as_float32(io.imread(os.path.join(path, "target.jpg")))
        source = img_as_float32(io.imread(os.path.join(path, "source.jpg")))
        
        out = {}
        out['driving'] = driving.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))

        return out 

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
