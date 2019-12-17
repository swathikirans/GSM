import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import sys

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, num_clips=1):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.num_clips = num_clips

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [
                    Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(self.root_path, directory.replace('frames', 'flow_x'),
                                            self.image_tmpl.format(idx).replace('image_', 'flow_x_'))).convert('L')
            y_img = Image.open(os.path.join(self.root_path, directory.replace('frames', 'flow_y'),
                                            self.image_tmpl.format(idx).replace('image_', 'flow_y_'))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=3]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.modality == 'Flow':
            num_frames = record.num_frames - 1
        else:
            num_frames = record.num_frames

        average_duration = (num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif num_frames - self.new_length + 1 > self.num_segments:
            offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if self.modality == 'Flow':
            num_frames = record.num_frames - 1
        else:
            num_frames = record.num_frames
        if num_frames > self.num_segments + self.new_length - 1:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        if self.modality == 'Flow':
            num_frames = record.num_frames - 1
        else:
            num_frames = record.num_frames

        tick = (num_frames - self.new_length + 1) / float(self.num_segments)

        if self.num_clips == 1:
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1

        elif self.num_clips == 2:
                offsets = [np.array([int(tick * x) for x in range(self.num_segments)])+1,
                           np.array([int(tick * x + tick / 2.0) for x in range(self.num_segments)]) + 1]
        return offsets



    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        # while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
        #     print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)))
        #     index = np.random.randint(len(self.video_list))
        #     record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        if self.num_clips > 1:
            process_data_final = []
            for k in range(self.num_clips):
                images = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    for i in range(self.new_length):
                        seg_imgs = self._load_image(record.path, p)
                        images.extend(seg_imgs)
                        if p < record.num_frames:
                            p += 1

                process_data, label = self.transform((images, record.label))
                process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)#
            return process_data_final, label

        else:
            images = list()
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

            process_data, label = self.transform((images, record.label))
            return process_data, label

    def __len__(self):
        return len(self.video_list)
