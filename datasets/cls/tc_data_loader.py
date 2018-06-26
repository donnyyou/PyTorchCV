"""
Batch Loader by Donny You
"""


import torch
import numpy as np
import json
from random import shuffle
import os


class TCDataLoader(object):

    def __init__(self, root_dir, batch_size):
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.item_list = self.__read_json_file(root_dir)
        self.idx = 0
        self.data_num = len(self.item_list)
        self.rnd_list = np.arange(self.data_num)
        shuffle(self.rnd_list)

    def num_data(self):
        return self.data_num

    def next_batch(self):
        batch_images = []
        batch_labels = []

        max_length = 50
        for i in range(self.batch_size):
            if self.idx != self.data_num:
                cur_idx = self.rnd_list[self.idx]
                im_path = self.item_list[cur_idx]['image_path']
                batch_images.append(np.load(os.path.join(self.root_dir, im_path)))
                batch_labels.append(int(self.item_list[cur_idx]['label']))
                cur_count = self.item_list[cur_idx]['word_count']
                max_length = max_length if max_length > cur_count else cur_count
                self.idx +=1
            else:
                self.idx = 0
                shuffle(self.rnd_list)
                cur_idx = self.rnd_list[self.idx]
                im_path = self.item_list[cur_idx]['image_path']
                batch_images.append(np.load(os.path.join(self.root_dir, im_path)))
                batch_labels.append(int(self.item_list[cur_idx]['label']))
                cur_count = self.item_list[cur_idx]['word_count']
                # max_length = max_length if max_length > cur_count else cur_count
                self.idx += 1

        for i in range(len(batch_images)):
            ori_arr = batch_images[i]
            now_arr = np.zeros((max_length, ori_arr.shape[1]))
            if max_length > ori_arr.shape[0]:
                now_arr[:ori_arr.shape[0], :] = ori_arr
            else:
                now_arr[:max_length, :] = ori_arr[:max_length, :]

            batch_images[i] = now_arr

        batch_images = np.array(batch_images).astype(np.float32)
        batch_labels = np.array(batch_labels).astype(np.int64)
        return torch.from_numpy(batch_images).unsqueeze(1), torch.from_numpy(batch_labels)

    def __read_json_file(self, root_dir):
        with open(os.path.join(root_dir, 'label.json'), 'r') as file_stream:
            items = json.load(file_stream)
            return items


if __name__ == "__main__":
    tc_data_loader = TCDataLoader('/home/donny/DataSet/Text/train', 3)
    images, labels = tc_data_loader.next_batch()
    print(images.unsqueeze(1).type())
    print(labels.shape)