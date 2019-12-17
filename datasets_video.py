import os
import torch
import torchvision
import torchvision.datasets as datasets


ROOT_DATASET = './dataset'


def return_something():
    root_data = 'something-v1/20bn-something-something-v1'
    filename_imglist_train = 'something-v1/train_videofolder.txt'
    filename_imglist_val = 'something-v1/val_videofolder.txt'
    prefix = '{:05d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_diving48():
    root_data = 'Diving48/frames'
    filename_imglist_train = 'Diving48/train_videofolder.txt'
    filename_imglist_val = 'Diving48/val_videofolder.txt'
    prefix = '{:05d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset):
    dict_single = {'something-v1': return_something, 'diving48':return_diving48}
    if dataset in dict_single:
            file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset]()
    else:
        raise ValueError('Unknown dataset '+dataset)
    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    root_data = os.path.join(ROOT_DATASET, root_data)

    return file_imglist_train, file_imglist_val, root_data, prefix
