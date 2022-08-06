import os
import re
import math
import linecache

import torch.utils.data as data
from pyknp import Juman
from sklearn.utils import shuffle

from argparse import ArgumentParser


# Juman++
jumanpp = Juman()

def tokenize(text):
    """
    You must input data at a one.
    """
    text = re.sub('\n', '', text)
    result = jumanpp.analysis(text)
    tokenized_text = [mrph.midasi for mrph in result.mrph_list()]

    return tokenized_text


class idDataLoader():
    def __init__(self, path, args:ArgumentParser):
        self.path = path
        self.all_length = 0
        self.each_length = []
        self.files = os.listdir(self.path)
        self.file_num = len(self.files)

        self.args = args

        self.__len__()

    def __call__(self, file_index, line_index):
        if file_index >= self.file_num:
            raise ValueError('index out of bounds.', file_index, self.file_num)

        result = ['','']
        file = self.files[file_index]
        file_len = self.each_length[file_index]

        if line_index > file_len:
            raise ValueError('index out of bounds.', line_index, file_len)

        f_path = os.path.join(self.path, file)
        line = linecache.getline(f_path, (line_index+1))
        linecache.clearcache()

        line = re.sub('\n', '', line)
        query_response = line.split(',')
        result[0] = query_response[0]
        result[1] = query_response[1]

        result[0] = result[0].split(' ')
        result[1] = result[1].split(' ')

        result[0] = [int(token_id) for token_id in result[0]]
        result[1] = [int(token_id) for token_id in result[1]]

        return result

    def __len__(self):
        if self.all_length != 0:
            return int(self.all_length/self.args.batch_size + 0.5)

        for file in self.files:
            f_path = os.path.join(self.path, file)
            f = open(f_path, 'r', encoding='utf-8')
            st = f.readline()
            f.close()
            tmp = int(st)
            if tmp < 0:
                tmp = 0
            self.each_length.append(tmp)
            self.all_length = self.all_length + tmp
            

        return math.ceil(self.all_length/self.args.batch_size)

class textDataset(data.Dataset):
    def __init__(self, dataloader, batch_size, make_mode=False, path=None):
        self.dataloader = dataloader
        self.file_num = self.dataloader.file_num
        self.each_length = self.dataloader.each_length
        self.batch_size = batch_size
        self.make_mode = make_mode

        self.path = path

        self.table = []
        self.generate_table()
        if not self.make_mode:
            self.next_epoch()
        self.current_dt = 0

    def __getitem__(self, idx):
        file_index = self.table[idx][0]
        line_index = self.table[idx][1]
        return self.dataloader(file_index, line_index)

    def __len__(self):
        length = math.ceil(len(self.table)/self.batch_size)
        return length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_dt == self.__len__():
            self.next_epoch()
            raise StopIteration()
        dts = [[],[]]
        b_size = 0
        if self.current_dt + 1 == self.__len__():
            b_size = len(self.table) % self.batch_size
        else:
            b_size = self.batch_size

        tmp = []
        for b in range(b_size):
            file_index = self.table[self.current_dt * self.batch_size + b][0]
            line_index = self.table[self.current_dt * self.batch_size + b][1]
            tmp = self.dataloader(file_index, line_index)
            dts[0].append(tmp[0])
            dts[1].append(tmp[1])

        if self.current_dt + 1 == self.__len__():
            dif = self.batch_size - b_size
            for _ in range(dif):
                dts[0].append([0])
                dts[1].append([0])

        self.current_dt += 1

        return dts

    def next_epoch(self):
        self.table = shuffle(self.table)
        self.current_dt = 0

    def generate_table(self):
        for i in range(self.file_num):
            for n in range(self.each_length[i]+1):
                if n==0:
                    continue
                tap = (i,n)
                self.table.append(tap)