for file in /data/vibhaa/nearest_neighbor_dataset/originals/*.mp4
do
    height=$(ffprobe -v error -show_entries \
	stream=height -of csv=p=0:s=x \
	"$file")

    fbname=$(basename "$file" .mp4)
    ffmpeg -y -i "$file" \
	-vf "crop=${height}:${height}" \
	-an "center_cropped/${fbname}_new2.mp4"
done

# carlsen video alone is off center
# -vf "crop=${height}:${height}:400:0" \
