#!/bin/sh
for i in 100 200 400 500 800 1000 
do
    # vp9
    if [[ $* == --vp9 ]]
    then
        ffmpeg -y -re -i ../samples/obama_shortened.webm \
            -c:v libvpx-vp9 -b:v ${i}k  -speed 8 -r 24 -deadline realtime\
            ../samples/obama_${i}k_vp9.webm
    fi

    # vp8
    if [[ $* == --vp8 ]]
    then
        ffmpeg -y -re -i ../samples/obama_shortened.webm \
            -c:v vp8 -b:v ${i}k  -speed 8 -r 24 -deadline realtime\
            ../samples/obama_${i}k_vp8.webm
    fi

    # h264
    if [[ $* == --h264 ]]
    then 
        ffmpeg -y -re -i ../samples/obama_shortened.mp4 \
            -c:v libx264 -b:v ${i}k  -preset ultrafast \
            ../samples/obama_${i}k.mp4
    fi
done

# vp9 and vp8 overshoot the target bandwidth, try using -crf instead to control them
