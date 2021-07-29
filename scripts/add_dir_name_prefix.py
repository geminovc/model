import glob
import pathlib
import os
import argparse

parser= argparse.ArgumentParser("Change name")
parser.add_argument('--directory',
        required=True,
        type=str,
        help='main directory containing files')

args = parser.parse_args()

files =  pathlib.Path(args.directory).glob('*')
files = sorted(['/'.join(str(seq).split('/')[-1:]) for seq in files])
dir_name = str(args.directory).split('/')[-1:][0]
if dir_name == '':
    dir_name = str(args.directory).split('/')[-2:][0]

for sub_file in files:
    print(sub_file)
    if os.path.isdir(args.directory + '/' + sub_file):
        print("won't change the name of sub directories.")
    else:
        source = args.directory + '/' + sub_file
        destination = args.directory + '/' + str(dir_name) + '_' +sub_file
        os.rename(source, destination)
        