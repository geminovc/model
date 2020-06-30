# downloads and temporally crops (into intervals where they are present/front-facing)
# the specified speakers
# from the sppech to gesture dataset

DIR="/data/vibhaa/speech2gesture-master/"
for speaker in "almaram" "angelica" "rock" "chemistry" "shelly"
do
    echo "SPEAKER: ${speaker}"
    python ${DIR}data/download/download_youtube.py \
        --base_path /data/vibhaa/speech2gesture-master/dataset \
        --speaker ${speaker}

    mkdir -p ${DIR}dataset/${speaker}/cropped

    python ${DIR}data/download/crop_intervals.py \
        --base_path /data/vibhaa/speech2gesture-master/dataset \
        --speaker ${speaker} \
        --output_path /data/vibhaa/speech2gesture-master/dataset/${speaker}/cropped
done
