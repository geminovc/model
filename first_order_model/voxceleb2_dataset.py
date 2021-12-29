import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from skimage.transform import resize

import numpy as np
from torch.utils.data import Dataset
from augmentation import AllAugmentationTransform

def read_video(name, frame_shape):
    """
    Read an mp4 video
    """
    
    video = np.array(mimread(name))
    if len(video.shape) == 3:
        video = np.array([gray2rgb(frame) for frame in video])
    if video.shape[-1] == 4:
        video = video[..., :3]

    video = np.array([resize(frame, frame_shape) for frame in video])
    video_array = img_as_float32(video)

    return video_array


class Voxceleb2Dataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
        - folder organized by person identifiers
        - further subdivided into sessions
        - and multiple video clips .mp4 files per session

    If session id is supplied, separated into test/train 
    for that session (for the corresponding person)

    If only person id is supplied, separate sessions of the person
    into test/train.
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True,
                    augmentation_params=None, person_id=None, 
                    train_percentage=0.75, session_id=None):
        self.root_dir = os.path.join(root_dir, "id" + str(person_id))
        self.frame_shape = tuple(frame_shape)
        self.person_id = person_id
        self.session_id = session_id

        train_videos = []
        test_videos = []
        
        # separate sessions into test/train and add all videos of a single
        # session either to test or train (no overlap)
        if self.session_id is None:
            print("Dataset is person", self.person_id, "from", self.root_dir) 
            sessions = os.listdir(self.root_dir)
            total_sessions = len(sessions)
            num_train_sessions = round(train_percentage * total_sessions)
        
            for i, session in enumerate(sessions):
                session_dir = os.path.join(self.root_dir, session)
                video_names = os.listdir(session_dir)
                session_videos = [os.path.join(session_dir, v) for v in video_names]
                if i < num_train_sessions:
                    train_videos.extend(session_videos)
                else:
                    test_videos.extend(session_videos)

        # separate videos of a single session into either test or train
        else:
            print("Dataset is person", self.person_id, "session", self.session_id)
            session_dir = os.path.join(self.root_dir, self.session_id)
            video_names = os.listdir(session_dir)
            session_videos = [os.path.join(session_dir, v) for v in video_names]
            num_train_videos = round(train_percentage * len(video_names))
            train_videos = session_videos[:num_train_videos]
            test_videos = session_videos[num_train_videos:]
        
        print("Test list", test_videos)
        print("Train list", train_videos)


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
        path = self.videos[idx]
        session_name = os.path.basename(os.path.dirname(path))
        video_name = session_name + "_" + os.path.basename(path)

        video_array = read_video(path, frame_shape=self.frame_shape)
        num_frames = len(video_array)
        frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
            num_frames)
        video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out
