# processing the raw data of the video datasets (Something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Bolei Zhou, Dec.2 2017
#
#

import os

dataset_name = 'something-something-v1'
root_dir = 'dataset/something-v1'
with open(os.path.join(root_dir, '%s-labels.csv'% 'something-something-v1')) as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
with open('category.txt','w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

files_input = ['%s-validation.csv'%'something-something-v1','%s-train.csv'%'something-something-v1']
files_output = ['val_videofolder.txt','train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(os.path.join(root_dir, filename_input)) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split(';')
        folders.append(items[0])
        idx_categories.append(dict_categories[items[1]])
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        dir_files = os.listdir(os.path.join(root_dir, '20bn-%s'%dataset_name, curFolder))
        output.append('%s %d %d'%(curFolder, len(dir_files), curIDX))
        print('%d/%d'%(i, len(folders)))
    with open(os.path.join(root_dir, filename_output),'w') as f:
        f.write('\n'.join(output))
