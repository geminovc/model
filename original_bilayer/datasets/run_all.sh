#!/bin/bash
command_to_run=${1}
dataset_path=${2}
save_path=${3}
tmux new-session -d -s my_session_0 "${command_to_run} && python test_on_video_dlib.py --gpu 0 --index 0 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_1 "${command_to_run} && python test_on_video_dlib.py --gpu 0 --index 1 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_2 "${command_to_run} && python test_on_video_dlib.py --gpu 0 --index 2 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_3 "${command_to_run} && python test_on_video_dlib.py --gpu 0 --index 3 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_4 "${command_to_run} && python test_on_video_dlib.py --gpu 0 --index 4 --root ${dataset_path} --save_path ${save_path}"

tmux new-session -d -s my_session_5 "${command_to_run} && python test_on_video_dlib.py --gpu 1 --index 5 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_6 "${command_to_run} && python test_on_video_dlib.py --gpu 1 --index 6 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_7 "${command_to_run} && python test_on_video_dlib.py --gpu 1 --index 7 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_8 "${command_to_run} && python test_on_video_dlib.py --gpu 1 --index 8 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_9 "${command_to_run} && python test_on_video_dlib.py --gpu 1 --index 9 --root ${dataset_path} --save_path ${save_path}"

tmux new-session -d -s my_session_10 "${command_to_run} && python test_on_video_dlib.py --gpu 2 --index 10 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_11 "${command_to_run} && python test_on_video_dlib.py --gpu 2 --index 11 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_12 "${command_to_run} && python test_on_video_dlib.py --gpu 2 --index 12 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_13 "${command_to_run} && python test_on_video_dlib.py --gpu 2 --index 13 --root ${dataset_path} --save_path ${save_path}"
tmux new-session -d -s my_session_14 "${command_to_run} && python test_on_video_dlib.py --gpu 2 --index 14 --root ${dataset_path} --save_path ${save_path}"
