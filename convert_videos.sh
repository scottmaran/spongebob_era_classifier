#1/bin/bash

#convert old and new videos
declare -a FolderArray=("spongebob_old_" "spongebob_new_")

#get number of files
function total_files {
	find $1 -type f | wc -l
}

function video_to_image {
	dir=$(pwd)
	local input_file=${dir}/${1}videos/${1}${2}.mov
	cd ${1}images
	ffmpeg -i $input_file  -r 0.5 -s 32x32 -qscale:v 2 ${1}${2}-%03d.jpg
	cd ..
}

#get number of videos for old and new
declare -i num_new_videos=$(total_files ${FolderArray[1]}videos)
declare -i num_old_videos=$(total_files ${FolderArray[0]}videos)

#for old spongebob
for ((c=1; c<=$num_old_videos; c++ )) do
	video_to_image ${FolderArray[0]} $c
done

echo "Finished with Old"
echo "Starting new"

#for new spongebob
 for ((c=1; c<=$num_new_videos; c++ )) do
	video_to_image ${FolderArray[1]} $c
done

echo "DONE!"
