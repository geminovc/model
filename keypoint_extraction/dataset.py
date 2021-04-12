
import torch
import pathlib
import cv2

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, sampling_rate, video_dir, sequences):
        super(Dataset).__init__()
        self.sampling_rate = sampling_rate
        self.sequences = sequences
        self.video_dir = video_dir
        self.seq_id = 0
    
    def fetch_vid_frames(self, cap):
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break

            if frame_num % self.sampling_rate == 0:
                yield frame , frame_num
            frame_num += 1

    def __iter__(self):
        while self.seq_id < len(self.sequences):
            filenames_vid = list((self.video_dir / self.sequences[self.seq_id]).glob('*'))
            self.seq_id += 1
            filenames_vid = [pathlib.Path(*filename.parts[-3:]).with_suffix('') for filename in filenames_vid]
            filenames = list(set(filenames_vid))
            filenames = sorted(filenames)
            for filename in filenames:
                video_path = pathlib.Path(self.video_dir) / filename.with_suffix('.mp4')
                name = str(filename).split('/')[len(str(filename).split('/'))-1]                                
                cap = cv2.VideoCapture(str(video_path))
                frame_num = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if frame is None:
                        cap.release()
                        break
                    if not ret:
                        cap.release()
                        break
                    if frame_num % self.sampling_rate == 0:
                        print("Gettin the frame ", frame_num , "from", str(video_path))
                        yield [frame , frame_num , str(filename),self.seq_id]
                    frame_num+=1
                cap.release()
    
                    #frame , frame_num = self.fetch_vid_frames(cap)
                    #yield frame , frame_num, filename





