import os
import json

dataset_name = 'Diving48'
root_dir = 'dataset'

files_input = ['%s_test.json'%dataset_name,'%s_train.json'%dataset_name]
files_output = ['val_videofolder.txt','train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(os.path.join(root_dir, dataset_name, filename_input), 'r') as f:
        data = json.load(f)
    output = []
    for i in range(len(data)):
        output.append('%s %d %d'%(data[i]['vid_name'], data[i]['end_frame'], data[i]['label']))
        print('%d/%d'%(i, len(data)))
    with open(os.path.join(root_dir, dataset_name, filename_output),'w') as f:
        f.write('\n'.join(output))
