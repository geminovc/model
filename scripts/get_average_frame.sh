# generates the average frame to drop a crop box on for each of the original
# videos for the specified speakers
# from the sppech to gesture dataset

DIR="/data/vibhaa/speech2gesture-master/"
for speaker in "oliver" # "almaram" "angelica" "rock" "chemistry" "shelly"
do
    mkdir -p ${DIR}dataset/${speaker}/averages
    for file in ${DIR}dataset/${speaker}/videos/*.mp4
    do
	fbname=$(basename "$file" .mp4)
        echo $fbname
        python get_average.py $file ${DIR}dataset/${speaker}/averages/${fbname}.jpg
    done
done
