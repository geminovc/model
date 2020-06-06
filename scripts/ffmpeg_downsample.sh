#!/bin/sh
for i in 50 100 200 400 500 800 1000
do
    ffmpeg -re -i ../samples/obama_shortened.mp4 -b:v ${i}K  -c:v vp9 -speed 8 -deadline realtime\
        ../samples/obama_${i}K.webm
done


