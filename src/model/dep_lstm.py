import torch
import torch.nn as nn
import pickle


class DependencyLSTMModel(nn.Module):
    def __init__(self, path, num_labels):
        super(DependencyLSTMModel, self).__init__()
        self.embeddings = self.__create_embeddings(path)
        self.embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(300, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, num_labels)

    def __create_embeddings(self, path):
        state_dict = pickle.load(open(path, "rb"))
        return nn.Embedding.from_pretrained(state_dict)

    def forward(self, input_ids):
        dep_input = input_ids[:, 0, :]
        x = self.embeddings(dep_input)
        out, (ht, ct) = self.lstm(x.float())
        out, _ = torch.max(out, 1)
        x = self.dropout(out)
        logits = self.classifier(x)
        return logits