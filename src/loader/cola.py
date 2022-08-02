import torch
from torch.utils.data import Dataset

import numpy as np
import pickle

import os


class CoLADataset(Dataset):
    def __init__(self, type_split_data, tokenizers, maxlen=512):
        self.data = self.__read_data(type_split_data)
        self.tokenizers = tokenizers
        self.pads = [tokenizer.vocab["[PAD]"] for tokenizer in self.tokenizers]
        self.maxlen = maxlen
        
    def __read_data(self, type_split_data):
        FOLDER_BERT = os.path.join("data", "dump_data_bert")
        FOLDER_DEPS = os.path.join("data", "dump_data_dep")
        X_bert = pickle.load(open(os.path.join(FOLDER_BERT, "data_cola.shuffled_clean_docs"), "rb"))
        X_deps = pickle.load(open(os.path.join(FOLDER_DEPS, "data_cola.shuffled_clean_docs"), "rb"))
        y = pickle.load(open(os.path.join(FOLDER_BERT, "data_cola.y"), "rb"))
        train_y = pickle.load(open(os.path.join(FOLDER_BERT, "data_cola.train_y"), "rb"))
        valid_y = pickle.load(open(os.path.join(FOLDER_BERT, "data_cola.valid_y"), "rb"))
        test_y = pickle.load(open(os.path.join(FOLDER_BERT, "data_cola.test_y"), "rb"))

        train_size = len(train_y)
        valid_size = len(valid_y)
        test_size = len(test_y)

        indexs = np.arange(0, len(X_bert))
        
        if type_split_data == "train":
            X_taken_bert = [X_bert[i] for i in indexs[:train_size]]
            X_taken_deps = [X_deps[i] for i in indexs[:train_size]]
            y_taken = [y[i] for i in indexs[:train_size]]
        
        elif type_split_data == "dev":
            X_taken_bert = [X_bert[i] for i in indexs[train_size : train_size + valid_size]]
            X_taken_deps = [X_deps[i] for i in indexs[train_size : train_size + valid_size]]
            y_taken = [y[i] for i in indexs[train_size : train_size + valid_size]]
        
        else:
            X_taken_bert = [X_bert[i] for i in indexs[train_size + valid_size : train_size + valid_size + test_size]]
            X_taken_deps = [X_deps[i] for i in indexs[train_size + valid_size : train_size + valid_size + test_size]]
            y_taken = [y[i] for i in indexs[train_size + valid_size : train_size + valid_size + test_size]]

        return {
            "text-bert": X_taken_bert,
            "text-deps": X_taken_deps,
            "deps": self.__read_deps(type_split_data),
            "label": y_taken
        }

    def __len__(self):
        return len(self.data["text"])
    
    def tokenize(self, text, maxlen=512):
        encoded = [tokenizer.encode(text) for tokenizer in self.tokenizers]
        for i in range(len(encoded)):
            len_pad = maxlen - len(encoded[i])
            if len_pad <= 0:
                encoded[i] = encoded[i][:maxlen]
            pad = [self.pads[i]] * len_pad
            encoded[i] += pad
        return torch.from_numpy(np.array(encoded))
    
    def __getitem__(self, idx):
        X, y = self.data["text"][idx], torch.from_numpy(np.array(self.data["label"][idx]))
        return self.tokenize(X), y