# generates the average frame to drop a crop box on for each of the original
# videos for the specified speakers
# from the sppech to gesture dataset

DIR="/data/vibhaa/speech2gesture-master/"
for speaker in "chemistry" "rock" "almaram" "angelica" "shelly" "jon" "oliver" "seth" "conan" "ellen"
do
    i=0
    mkdir -p ${DIR}dataset/${speaker}/averages
    for file in ${DIR}dataset/${speaker}/videos/*
    do
        filename=$(basename "$file")
        fbname=${filename%%.*}
        echo $fbname
        python get_average.py $file ${DIR}dataset/${speaker}/averages/${fbname}.jpg
    done
done
