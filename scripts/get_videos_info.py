"""
This script records the following information for videos in --video_root:

    'video_path','width','height', 'num_frames','avg_frame_rate', 'bit_rate','duration'

Inputs
----------

video_root: root to the video directory. The videos should be stored in voxceleb2 format.
result_file_name: name of csv file to write results out to


Outputs
----------

The output is a csv file called result_file_name that contains the coulmns: 'video_path','width','height', 'num_frames','avg_frame_rate', 'bit_rate','duration', 'pix_fmt'

Sample usage:

python get_videos_info.py --video_root '/path/to/videos' --result_file_name results.csv


"""

from __future__ import unicode_literals, print_function
import argparse
import ffmpeg
import sys
import glob
import pathlib

parser = argparse.ArgumentParser(description='Get video information')
parser.add_argument('--video_root',
                        type=str,
                        default = '/video-conf/vedantha/voxceleb2/dev/mp4',
                        help='root directory of the videos')

parser.add_argument('--result_file_name',
                       type=str,
                       required=True,
                       help='name of csv file to write results out to')


def get_video_info (in_filename):
    try:
        probe = ffmpeg.probe(in_filename)
    except ffmpeg.Error as e:
        print(e.stderr, file=sys.stderr)

    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found', file=sys.stderr)

    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])
    bit_rate = int(video_stream['bit_rate'])
    duration = video_stream['duration']
    avg_frame_rate = video_stream['avg_frame_rate']
    pix_fmt = video_stream['pix_fmt']
    
    print('width: {} , height: {}, num_frames: {}, avg_frame_rate: {}, bit_rate: {}, duration: {} s, pix_fmt: {}'.format(width, height, num_frames, avg_frame_rate, bit_rate, duration, pix_fmt))

    return width, height, num_frames, avg_frame_rate, bit_rate, duration, pix_fmt

if __name__ == '__main__':
    args = parser.parse_args()
    video_paths = pathlib.Path(args.video_root).glob('*/*/*')
    count = 0


    # write to csv
    with open(args.result_file_name, 'w') as f:
        f.write("%s,%s,%s,%s,%s,%s,%s,%s\n"%('video_path','width','height', 'num_frames', 
                'avg_frame_rate', 'bit_rate','duration', 'pix_fmt'))

        for video_path in video_paths:
            count+=1
            try:
                width, height, num_frames, avg_frame_rate, bit_rate, duration, pix_fmt = get_video_info (video_path)
                f.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (video_path, width, height, num_frames, 
                    avg_frame_rate, bit_rate, duration, pix_fmt))
            except:
                print("Exception happend in getting the information of", video_path)
    
    print("total number of videos", count)