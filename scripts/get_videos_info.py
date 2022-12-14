"""
This script records the following information for videos in --video_root:

    'video_path','width','height', 'num_frames','avg_frame_rate', 'bit_rate','duration' 'pix_fmt', 
    'crf', 'chroma_qp_offset', 'qpmin', 'qpmax', 'qpstep'

Inputs
----------

video_root: root to the video directory. The videos should be stored in voxceleb2 format.
csv_file_name: name of csv file to write results out to
hist_file_name: name of png file to write the histogram out to
bit_rate_threshold: threshold of acceptable bit rates

Outputs
----------

The output is a csv file called csv_file_name that contains the coulmns: 

'video_path','width','height', 'num_frames','avg_frame_rate', 'bit_rate','duration', 'pix_fmt',
'crf', 'chroma_qp_offset', 'qpmin', 'qpmax', 'qpstep'

Sample usage:

python get_videos_info.py --video_root '/path/to/videos' --csv_file_name /path/to/save/results.csv --hist_file_name /path/to/save/hist.png \
--bit_rate_threshold 500000


"""

from __future__ import unicode_literals, print_function
import argparse
import ffmpeg
import sys
import glob
import os
import pathlib

parser = argparse.ArgumentParser(description='Get video information')
parser.add_argument('--video_root',
                        type = str,
                        default = '/video-conf/vedantha/voxceleb2/dev/mp4',
                        help = 'root directory of the videos')

parser.add_argument('--csv_file_name',
                       type = str,
                       required = True,
                       help = 'name of csv file to write results out to')

parser.add_argument('--hist_file_name',
                       type = str,
                       required = True,
                       help = 'name of png file to write the histogram out to')

parser.add_argument('--bit_rate_threshold',
                       type = int,
                       default = 480000,
                       help = 'threshold of acceptable bit rates')

# Outputs the 'width','height', 'num_frames','avg_frame_rate', 'bit_rate','duration', 'pix_fmt', 'crf',
# 'chroma_qp_offset', 'qpmin', 'qpmax', 'qpstep' for the video in in_filename
def get_video_info (in_filename):
    try:
        probe = ffmpeg.probe(in_filename)
    except ffmpeg.Error as e:
        print(e.stderr, file = sys.stderr)

    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found', file=sys.stderr)
        width, height, num_frames, avg_frame_rate, bit_rate, duration, pix_fmt, crf, chroma_qp_offset, qpmin, qpmax, qpstep = '0', '0', '0', '0', '0', '0', \
                                                                                                                              'None', '0', '0', '0', '0', '0'
    else:
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        num_frames = int(video_stream['nb_frames'])
        bit_rate = int(video_stream['bit_rate'])
        duration = video_stream['duration']
        avg_frame_rate = video_stream['avg_frame_rate']
        pix_fmt = video_stream['pix_fmt']
        try:
            mediainfo_crf = os.popen("mediainfo \"%s\" | grep \"Encoding settings\" | cut -d':' -f2- | tr '/' '\n' | sed 's/ //' | grep crf= " % (str(in_filename))).read()
            crf = (mediainfo_crf.split('=')[1]).split('\n')[0]
            mediainfo_chroma_qp_offset = os.popen("mediainfo \"%s\" | grep \"Encoding settings\" | cut -d':' -f2- | tr '/' '\n' | sed 's/ //' | grep chroma_qp_offset= " % (str(in_filename))).read()
            chroma_qp_offset = (mediainfo_chroma_qp_offset.split('=')[1]).split('\n')[0]
            mediainfo_qpmin = os.popen("mediainfo \"%s\" | grep \"Encoding settings\" | cut -d':' -f2- | tr '/' '\n' | sed 's/ //' | grep qpmin= " % (str(in_filename))).read()
            qpmin = (mediainfo_qpmin.split('=')[1]).split('\n')[0]
            mediainfo_qpmax = os.popen("mediainfo \"%s\" | grep \"Encoding settings\" | cut -d':' -f2- | tr '/' '\n' | sed 's/ //' | grep qpmax= " % (str(in_filename))).read()
            qpmax = (mediainfo_qpmax.split('=')[1]).split('\n')[0]
            mediainfo_qpstep = os.popen("mediainfo \"%s\" | grep \"Encoding settings\" | cut -d':' -f2- | tr '/' '\n' | sed 's/ //' | grep qpstep= " % (str(in_filename))).read()
            qpstep = (mediainfo_qpstep.split('=')[1]).split('\n')[0]
        except: 
            crf, chroma_qp_offset, qpmin, qpmax, qpstep = 100, 100, 100, 100, 100

    print('width: {} , height: {}, num_frames: {}, avg_frame_rate: {}, bit_rate: {}, duration: {} s, pix_fmt: {}, crf: {} '.format(width,
                                                                                                                            height, num_frames,
                                                                                                                            avg_frame_rate, bit_rate,
                                                                                                                            duration, pix_fmt, crf))
    print('chroma_qp_offset: {}, qpmin: {}, qpmax: {}, qpstep: {}'.format(chroma_qp_offset, qpmin, qpmax, qpstep))
    return width, height, num_frames, avg_frame_rate, bit_rate, duration, pix_fmt, crf, chroma_qp_offset, qpmin, qpmax, qpstep

# Finds the element in an array with the most frequency of appearance in the array
def get_most_frequent_element(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency > counter):
            counter = curr_frequency
            num = i
  
    return num, counter


if __name__ == '__main__':
    args = parser.parse_args()
    video_paths = pathlib.Path(args.video_root).glob('*/*/*')
    count = 0
    # write to csv
    with open(args.cvs_file_name, 'w') as f:
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%('video_path','width','height', 'num_frames', 
                'avg_frame_rate', 'bit_rate','duration', 'pix_fmt','crf', 'chroma_qp_offset', 'qpmin', 'qpmax', 'qpstep'))

        for video_path in video_paths:
            count += 1
            try:
                width, height, num_frames, avg_frame_rate, bit_rate, duration, pix_fmt, crf, chroma_qp_offset, qpmin, qpmax, qpstep = get_video_info (video_path)
                f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (video_path, width, height, num_frames, 
                    avg_frame_rate, bit_rate, duration, pix_fmt, crf, chroma_qp_offset, qpmin, qpmax, qpstep))
            except:
                print("Exception happend in getting the information of", video_path)
    
    print("Process the output results")    
    data_dict = {}
    cvs_file_name = str(args.csv_file_name)    
    with open(cvs_file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            count += 1
            data_dict[row['video_path']] = row['bit_rate']
    
    print("Number of videos", count)
    bitrate_values = [int(v) for v in data_dict.values() ]

    # Save bit rate histogram
    plt.figure()
    plt.hist(bitrate_values, bins=30)  
    plt.savefig(str(args.hist_file_name))
    
    # Find the sequences with the most number of HQ videos (videos with bit_rate > bit_rate_threshold) 
    print("minimum bit_rate", min(bitrate_values))
    HQ_videos = ['/'.join(str(k).split('/')[-3:]) for k, v in data_dict.items() if int(v) >= int(args.bit_rate_threshold)]
    HQ_ids = [str(k).split('/')[0] for k in HQ_videos]
    most_frequent_id, freq = get_most_frequent_element(HQ_ids)
    print("Most frequent id and freq:", most_frequent_id , freq)
    desired_sequences = [k for k in HQ_videos if str(k).split('/')[0] == most_frequent_id]
    print("desired sequences:", desired_sequences)