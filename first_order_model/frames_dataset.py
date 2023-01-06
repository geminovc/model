import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
import imageio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from first_order_model.augmentation import AllAugmentationTransform
import glob
import av

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


def read_single_frame(filename):
    """ read a single png file into a numpy array """
    image = io.imread(filename)
    return img_as_float32(image)


def get_num_frames(filename):
    cmd = f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format csv {filename}"
    num_frames = os.popen(cmd).read()
    num_frames = int(num_frames.split(',')[1])
    return num_frames


def get_frame(filename, frame_num, ifnormalize=True):
    reader = imageio.get_reader(filename, "ffmpeg")
    reader.set_image_index(frame_num)
    frame = np.array(reader.get_next_data())
    if ifnormalize:
        frame = img_as_float32(frame)
    reader.close()
    return frame


def get_video_details(filename):
    container = av.open(file=filename, format=None, mode='r')
    fps = container.streams.video[0].average_rate
    video_stream = container.streams.video[0]
    nr, dr = video_stream.time_base.as_integer_ratio()
    container.close()
    return fps, nr, dr

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

        if person_id is not None:
            root_dir = os.path.join(root_dir, person_id)
            self.root_dir = root_dir
        
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            print("number of train videos", len(train_videos))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
             
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
        if (self.is_train and self.id_sampling):
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        elif self.is_train:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
            path_suffix = name.split("_")[-1]

            # get a different clip of the same larger video to pull target frame from
            if self.person_id != 'generic':
                path_tgt = np.random.choice(glob.glob(os.path.join(self.root_dir, "*_" + path_suffix)))
            else:
                path_tgt = path
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)
        av_video_array = []

        if self.is_train:
            if  os.path.isdir(path):
                frames = os.listdir(path)
                num_frames = len(frames)
                frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
                video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            elif path.split('.')[-1] == 'mp4':
                num_frames_src = get_num_frames(path)
                frame_idx_src = np.random.choice(num_frames_src - 1, replace=True) 
                
                num_frames_tgt = get_num_frames(path_tgt)
                frame_idx_tgt = np.random.choice(num_frames_tgt - 1, replace=True) 
                try:
                    video_array = np.array([get_frame(path, frame_idx_src), get_frame(path_tgt, frame_idx_tgt)])
                    fps, time_base_nr, time_base_dr = get_video_details(path_tgt)
                except:
                    print("Couldn't get indices", frame_idx, "of video", path, "with", num_frames, "total frames")
            else:
                src_frame = read_single_frame(path)
                tgt_frame_path = os.path.join(self.root_dir, np.random.choice(os.listdir(self.root_dir)))
                tgt_frame = read_single_frame(tgt_frame_path)
                video_array = [src_frame, tgt_frame] if self.is_train else [src_frame]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['time_base_nr'] = time_base_nr
            out['time_base_dr'] = time_base_dr
        else:
            out['video_path'] = str(path)
        
        out['name'] = video_name

        return out


class VP9Dataset(Dataset):
    """
        Load data from Vp9 pre-encoded videos by separating the low-res video from the ground truth.
    """
    
    def __init__(self, gt_root_dir, lr_root_dir, resolution, bitrate, frame_shape=(256, 256, 3), is_train=True,
                 random_seed=0, augmentation_params=None, person_id=None):
        self.gt_root_dir = gt_root_dir
        self.lr_root_dir = lr_root_dir
        self.frame_shape = tuple(frame_shape)
        self.person_id = person_id
        self.bitrate = bitrate

        if person_id is not None:
            self.gt_root_dir = os.path.join(self.gt_root_dir, person_id)
            self.lr_root_dir = os.path.join(self.lr_root_dir, person_id, f'{resolution}')
        
        assert os.path.exists(os.path.join(self.lr_root_dir, 'train'))
        assert os.path.exists(os.path.join(self.lr_root_dir, 'test'))
        print("Use predefined train-test split.")
        
        gt_train_videos = os.listdir(os.path.join(self.gt_root_dir, 'train'))
        print("number of train videos", len(gt_train_videos))

        # lr_test_videos = [t for t in test_videos if t.split('_')[-1].split('K')[0] == bitrate]
        gt_test_videos = os.listdir(os.path.join(self.gt_root_dir, 'test'))
        print("number of test videos", len(gt_test_videos))
         
        self.lr_root_dir = os.path.join(self.lr_root_dir, 'train' if is_train else 'test')
        self.gt_root_dir = os.path.join(self.gt_root_dir, 'train' if is_train else 'test')

        if is_train:
            self.videos = gt_train_videos
        else:
            self.videos = gt_test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train:
            name = self.videos[idx]
            gt_path = os.path.join(self.gt_root_dir, name)
            gt_path_suffix = name.split("_")[-1]
            lr_path = os.path.join(self.lr_root_dir, f'{name}_{self.bitrate}K.webm')

            # get a different clip of the same larger video to pull target frame from
            if self.person_id != 'generic':
                gt_path_tgt = np.random.choice(glob.glob(os.path.join(self.gt_root_dir, "*_" + gt_path_suffix)))
                tgt_name = os.path.basename(gt_path_tgt)
                lr_path_tgt = os.path.join(self.lr_root_dir, f'{tgt_name}_{self.bitrate}K.webm')
            else:
                gt_path_tgt = gt_path
                lr_path_tgt = lr_path
        else:
            name = self.videos[idx]
            gt_path = os.path.join(self.gt_root_dir, name)

        video_name = os.path.basename(gt_path)
        av_video_array = []

        if self.is_train:
            assert gt_path.split('.')[-1] == 'mp4'
            num_frames_src = get_num_frames(gt_path)
            frame_idx_src = np.random.choice(num_frames_src - 1, replace=True) 
            
            num_frames_tgt = get_num_frames(gt_path_tgt)
            frame_idx_tgt = np.random.choice(num_frames_tgt - 1, replace=True) 
            try:
                gt_video_array = np.array([get_frame(gt_path, frame_idx_src), get_frame(gt_path_tgt, frame_idx_tgt)])
                lr_video_array = np.array([get_frame(lr_path, frame_idx_src), get_frame(lr_path_tgt, frame_idx_tgt)])
            except:
                print("Couldn't get indices", frame_idx, "of video", path, "with", num_frames, "total frames")

        if self.transform is not None:
            gt_video_array = self.transform(gt_video_array)
            lr_video_array = self.transform(lr_video_array)

        out = {}
        if self.is_train:
            source = np.array(gt_video_array[0], dtype='float32')
            driving = np.array(gt_video_array[1], dtype='float32')
            driving_lr = np.array(lr_video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['driving_lr'] = driving_lr.transpose((2, 0, 1))
        else:
            out['gt_video_path'] = str(gt_path)
            out['lr_video_path'] = str(lr_path)

        out['name'] = video_name

        return out


class MetricsDataset(Dataset):
    """
        Load a select set of frames for computing consistent metrics/visuals on
    """

    def __init__(self, root_dir, frame_shape, person_id=None):
        self.root_dir = root_dir
        if person_id is not None:
            root_dir = os.path.join(root_dir, person_id, "validation")
            self.root_dir = root_dir
        
        self.frame_shape = tuple(frame_shape)
        self.videos = os.listdir(root_dir)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        file_name = self.videos[idx]
        path = os.path.join(self.root_dir, file_name)
        assert os.path.isdir(path)

        driving = img_as_float32(io.imread(os.path.join(path, "target.png")))
        source = img_as_float32(io.imread(os.path.join(path, "source.png")))
        
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
