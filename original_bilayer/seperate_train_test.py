import os
import shutil
import  pathlib
data_root = "/data/pantea/video_conf/minidataset/"

# Data paths
imgs_dir = data_root + 'imgs'  
imgs_dir_train = data_root + 'imgs/train/' 
imgs_dir_test = data_root + 'imgs/test/' 

poses_dir = data_root + 'keypoints'  
poses_dir_train = data_root + 'keypoints/train/' 
poses_dir_test = data_root + 'keypoints/test/'

segs_dir = data_root + 'segs'  
segs_dir_train = data_root + 'segs/train/' 
segs_dir_test = data_root + 'segs/test/' 

# person_id sequences list
personId_sequences = pathlib.Path(imgs_dir_train).glob('*')
personId_sequences = ['/'.join(str(seq).split('/')[-1:]) for seq in personId_sequences]
personId_sequences = sorted(personId_sequences)

print(personId_sequences)

print("Moving the first 20 percent of people you will never see while training -------")

not_seen_persons_sequences = personId_sequences[:int(0.1*len(personId_sequences))]

for person_id in not_seen_persons_sequences:
    
    # images
    source = imgs_dir_train + person_id
    destination = imgs_dir_test 
    os.makedirs(destination, exist_ok=True)
    #print("moving ",source," to " ,destination)
    dest = shutil.move(source, destination) 

    # keypoints
    source = poses_dir_train + person_id
    destination = poses_dir_test 
    os.makedirs(destination, exist_ok=True)
    #print("moving ",source," to " ,destination)
    dest = shutil.move(source, destination) 

    # segs
    source = segs_dir_train + person_id
    destination = segs_dir_test 
    os.makedirs(destination, exist_ok=True)
    #print("moving ",source," to " ,destination)
    dest = shutil.move(source, destination) 

seen_persons_sequences = personId_sequences[int(0.1*len(personId_sequences)):]

for person_id in seen_persons_sequences:
    videoId_sequences = pathlib.Path(imgs_dir_train + person_id).glob('*')
    videoId_sequences = ['/'.join(str(seq).split('/')[-1:]) for seq in videoId_sequences]
    videoId_sequences = sorted(videoId_sequences)
    not_seen_videoId_sequences = videoId_sequences[:int(0.1*len(videoId_sequences))]
    seen_videoId_sequences = videoId_sequences[int(0.1*len(videoId_sequences)):]
    for video_id in not_seen_videoId_sequences:
    
        # images
        source = imgs_dir_train + person_id + "/" + video_id
        destination = imgs_dir_test + person_id 
        os.makedirs(destination, exist_ok=True)
        #print("moving ",source," to " ,destination)
        dest = shutil.move(source, destination) 

        # keypoints
        source = poses_dir_train + person_id + "/" + video_id
        destination = poses_dir_test + person_id 
        os.makedirs(destination, exist_ok=True)
        #print("moving ",source," to " ,destination)
        dest = shutil.move(source, destination) 

        # segs
        source = segs_dir_train + person_id + "/" + video_id
        destination = segs_dir_test + person_id 
        os.makedirs(destination, exist_ok=True)
        #print("moving ",source," to " ,destination)
        dest = shutil.move(source, destination) 
    
    for video_id in seen_videoId_sequences:
        seqId_sequences = pathlib.Path(imgs_dir_train + person_id +"/"+ video_id).glob('*')
        seqId_sequences = ['/'.join(str(seq).split('/')[-1:]) for seq in seqId_sequences]
        seqId_sequences = sorted(seqId_sequences)
        if len(seqId_sequences) > 5:
            not_seen_seqId_sequences = seqId_sequences[:int(0.15*len(seqId_sequences))]
            for seq_id in not_seen_seqId_sequences:

                # images
                source = imgs_dir_train + person_id + "/" + video_id + "/" + seq_id
                destination = imgs_dir_test + person_id + "/" + video_id
                os.makedirs(destination, exist_ok=True)
                #print("moving ",source," to " ,destination)
                dest = shutil.move(source, destination) 

                # keypoints
                source = poses_dir_train + person_id + "/" + video_id + "/" + seq_id
                destination = poses_dir_test + person_id + "/" + video_id
                os.makedirs(destination, exist_ok=True)
                #print("moving ",source," to " ,destination)
                dest = shutil.move(source, destination) 

                # segs
                source = segs_dir_train + person_id + "/" + video_id + "/" + seq_id
                destination = segs_dir_test + person_id + "/" + video_id
                os.makedirs(destination, exist_ok=True)
                #print("moving ",source," to " ,destination)
                dest = shutil.move(source, destination) 
        
