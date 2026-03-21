import h5py
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
import numpy as np
import os

class TrainDataset(Dataset):
    text_cache = None
    writer_cache = None

    def __init__(self, hdf5_file, text_file, writer_file, transform=None):
        self.hdf5_file = hdf5_file
        self.transform = transform

        print(f'Initializing dataset from {hdf5_file}')
        self.hf = h5py.File(hdf5_file, 'r')

        self.keys = []
        self.lengths = []

        for key in self.hf.keys():
            num_points = len(self.hf[key]['point_seq'])
            if 2000>= num_points >= 200:
                self.keys.append(key)
                self.lengths.append(num_points)

        # self.sorted_indices = np.argsort(self.lengths)
        self.sorted_indices = np.argsort(self.lengths)[::-1]
        self.len = len(self.keys)

        if TrainDataset.text_cache is None and text_file is not None:
            TrainDataset.text_cache = self.read_all_chars(text_file)
        
        if TrainDataset.writer_cache is None and writer_file is not None:
            TrainDataset.writer_cache = self.read_all_writer(writer_file)

    def read_all_chars(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        decoded_dict = {}
        for idx, key in enumerate(data.keys()):
            decoded_key = key
            decoded_dict[decoded_key] = idx

        return decoded_dict
    
    def read_all_writer(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        wirter_dict = {}
        for idx, key in enumerate(data['train']):
            wirter_dict[key] = idx

        return wirter_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sorted_idx = self.sorted_indices[idx]

        key = self.keys[sorted_idx]
        writer_id = self.hf[key]['writer_id'][()].decode('utf-8')
        point_seq = np.array(self.hf[key]['point_seq'][:], dtype=np.float32)
        char_points_idx = self.hf[key]['char_points_idx'][:]
        line_text = self.hf[key]['line_text'][()].decode('utf-8')

        if self.transform:
            point_seq = self.transform(point_seq)
        
        point_seq = torch.tensor(point_seq, dtype=torch.float32)

        return writer_id, point_seq, line_text, char_points_idx

    @staticmethod
    def get_text_index(line_text):
        font_data = TrainDataset.text_cache

        char_idx = []
        for string in line_text:
            converted = []
            for char in string:
                if char not in font_data:
                    raise ValueError(f"未知字符: {char}")
                converted.append(font_data[char])
            char_idx.append(converted)

        text_tensor = [torch.tensor(lst, dtype=torch.long) for lst in char_idx]
        padded_text = pad_sequence(text_tensor, batch_first=True, padding_value=-1)

        return padded_text
    
    @staticmethod
    def change_text_index(line_text):
        font_data = TrainDataset.text_cache

        char_idx = []
        for string in line_text:
            converted = []
            for char in string:
                if char not in font_data:
                    raise ValueError(f"未知字符: {char}")
                converted.append(font_data[char])
            new_converted = modify_suffix_ids_only(converted, vocab = list(font_data.values()))
            char_idx.append(new_converted)

        text_tensor = [torch.tensor(lst, dtype=torch.long) for lst in char_idx]
        padded_text = pad_sequence(text_tensor, batch_first=True, padding_value=-1)

        return padded_text
    
    @staticmethod
    def get_writer_index(writer_id):
        writer_cache = TrainDataset.writer_cache

        if writer_cache is not None:
            writer_index_map = []
            for i in writer_id:
                writer_index_map.append(writer_cache[i])
        
            writer_index = torch.tensor(writer_index_map, dtype=torch.long)
            return writer_index
        
        else:
            return None

    @staticmethod
    def collate_fn(batch):
        writer_id, sequences, line_text, char_points_idx = zip(*batch)
        
        text_index = TrainDataset.get_text_index(line_text)
        writer_index = TrainDataset.get_writer_index(writer_id)

        max_length = max([seq.shape[0] for seq in sequences])
        min_length = min([seq.shape[0] for seq in sequences])

        while max_length % 8 != 0:
            max_length += 1

        custom_padding_value = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)

        padded_sequences = []
        for seq in sequences:
            if seq.shape[0] < max_length:
                extra_padding = custom_padding_value.repeat(max_length - seq.shape[0], 1)
                padded_seq = torch.cat([seq, extra_padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        batch_tensor = torch.stack(padded_sequences)

        is_padding = (batch_tensor == custom_padding_value).all(dim=-1)
        mask = (~is_padding)

        return batch_tensor, mask, text_index, char_points_idx, writer_index

class ValDataset(Dataset):
    text_cache = None

    def __init__(self, hdf5_file, text_file, writer_file, transform=None):
        self.hdf5_file = hdf5_file
        self.transform = transform

        print(f'Initializing dataset from{hdf5_file}')

        self.hf = h5py.File(hdf5_file, 'r')

        self.keys = []
        self.lengths = []

        for key in self.hf.keys():
            num_points = len(self.hf[key]['point_seq'])
            if num_points >= 200:
                self.keys.append(key)
                self.lengths.append(num_points)

        # self.sorted_indices = np.argsort(self.lengths)
        self.sorted_indices = np.argsort(self.lengths)[::-1]
        self.len = len(self.keys)

        if ValDataset.text_cache is None and text_file is not None:
            ValDataset.text_cache = self.read_all_chars(text_file)

    def read_all_chars(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        decoded_dict = {}
        for idx, key in enumerate(data.keys()):
            decoded_key = key
            decoded_dict[decoded_key] = idx

        return decoded_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sorted_idx = self.sorted_indices[idx]

        key = self.keys[sorted_idx]
        writer_id = self.hf[key]['writer_id'][()].decode('utf-8')
        point_seq = np.array(self.hf[key]['point_seq'][:], dtype=np.float32)
        char_points_idx = self.hf[key]['char_points_idx'][:]
        line_text = self.hf[key]['line_text'][()].decode('utf-8')

        if self.transform:
            point_seq = self.transform(point_seq)
        
        point_seq = torch.tensor(point_seq, dtype=torch.float32)

        return writer_id, point_seq, line_text, char_points_idx

    @staticmethod
    def get_text_index(line_text):
        font_data = ValDataset.text_cache

        char_idx = []
        for string in line_text:
            converted = []
            for char in string:
                if char not in font_data:
                    raise ValueError(f"未知字符: {char}")
                converted.append(font_data[char])
            converted.append(font_data['、']) # one more char
            char_idx.append(converted)

        text_tensor = [torch.tensor(lst, dtype=torch.long) for lst in char_idx]
        padded_text = pad_sequence(text_tensor, batch_first=True, padding_value=-1)

        return padded_text

    @staticmethod
    def collate_fn(batch):
        writer_id, sequences, line_text, char_points_idx = zip(*batch)

        text_index = ValDataset.get_text_index(line_text)

        max_length = max([seq.shape[0] for seq in sequences])
        min_length = min([seq.shape[0] for seq in sequences])

        while max_length % 8 != 0:
            max_length += 1

        custom_padding_value = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)

        padded_sequences = []
        for seq in sequences:
            if seq.shape[0] < max_length:
                extra_padding = custom_padding_value.repeat(max_length - seq.shape[0], 1)
                padded_seq = torch.cat([seq, extra_padding], dim=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        batch_tensor = torch.stack(padded_sequences)

        is_padding = (batch_tensor == custom_padding_value).all(dim=-1)
        mask = (~is_padding)

        return batch_tensor, mask, text_index, char_points_idx, writer_id