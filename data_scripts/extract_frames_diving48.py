import os
import glob

src_dir = 'dataset/Diving48/rgb' # Dir containing the videos
des_dir = 'dataset/Diving48/frames' # Output dir to save the videos

vid_files = glob.glob1(src_dir, '*.mp4')
for vid in vid_files:
    des_dir_path = os.path.join(des_dir, vid[:-4])
    if not os.path.exists(des_dir_path):
        os.makedirs(des_dir_path)
    os.system('ffmpeg -i ' + os.path.join(src_dir, vid) + ' -qscale:v 2 ' + des_dir_path + '/frames%05d.jpg')
