# re-encodes videos of the specified spakers at lower bitrates 
# for the speech2gesture dataset

DIR="/data/vibhaa/speech2gesture-master/"
for speaker in "conan" "chemistry" "ellen" "oliver" "seth" "almaram" "angelica" "rock" "shelly"
do
    echo "SPEAKER: ${speaker}"
    mkdir -p ${DIR}dataset/${speaker}/downsampled
    for file in ${DIR}dataset/${speaker}/spatially_cropped/*.mp4
    do
	fbname=$(basename "$file" .mp4)
	for i in 50 100 200 500 1000
	do
           ffmpeg -y -re -i "$file" \
            -c:v libx264 -b:v ${i}k  -preset ultrafast \
            -c:a copy "${DIR}dataset/${speaker}/downsampled/${fbname}_${i}Kb.mp4"
	done
    done
done
