import torch
import torch.nn as nn
import pickle
import numpy as np


class DependencyLSTMLocalModel(nn.Module):
    def __init__(self, path_word_emb, path_dep_emb, num_labels):
        super(DependencyLSTMLocalModel, self).__init__()
        self.word_embeddings = self.__create_embeddings(path_word_emb)
        self.dep_embeddings = self.__create_embeddings(path_dep_emb)
        self.word_embeddings.weight.requires_grad = False
        self.dep_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(300, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, num_labels)
        self.device = "cuda:3"

    def add_special_ids(self, unk_id, pad_id):
        self.pad_id = pad_id
        self.skip_ids = set([self.pad_id, unk_id])
        
    def __create_embeddings(self, path):
        state_dict = pickle.load(open(path, "rb"))
        return nn.Embedding.from_pretrained(state_dict)

    def __mask(self, ids):
        mask = []
        for id_ in ids:
            if int(id_.cpu().detach().numpy()) not in self.skip_ids:
                mask.append(1)
            else:
                mask.append(0)
        return torch.from_numpy(np.array(mask).reshape(-1, 1)).to(self.device)

    def __is_skip(self, ids):
        for id_ in ids:
            if int(id_.cpu().detach().numpy()) not in self.skip_ids:
                return False
        return True

    def forward(self, word_ids, deps_ids):
        word_inputs = word_ids[:, 1, :]
        deps_inputs = deps_ids

        word_embs = self.word_embeddings(word_inputs)
        dep_embs = self.dep_embeddings(deps_inputs)

        for i, batch_deps in enumerate(deps_ids):
            for j, deps_id in enumerate(batch_deps):
                mask = self.__mask(deps_id)
                sum_ = mask.sum()
                if sum_:
                    taken_emb = (dep_embs[i][j] * mask).sum(axis=0)
                    word_embs[i][j] = 0.5 * word_embs[i][j] + 0.5 * taken_emb / sum_

        out, (ht, ct) = self.lstm(word_embs.float())
        out, _ = torch.max(out, 1)
        x = self.dropout(out)
        logits = self.classifier(x)
        return logits