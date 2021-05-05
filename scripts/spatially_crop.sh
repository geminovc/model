# uses a pickle file per celebrity with annotations on what
# square to cut and cuts all the input "temporally cropped" videos
# to that square, essentially spatially cropping them also

DIR="/data/vibhaa/speech2gesture-master/"
for speaker in "chemistry" "rock" "almaram" "angelica" "shelly" "jon" "oliver" "seth" "conan" "ellen"
do
    python3 crop_coordinates.py \
        --data-dir ${DIR}dataset/${speaker} \
        --pickle-name /${DIR}dataset/averages/annotations_for_all_celebs/${speaker}.pkl
done
