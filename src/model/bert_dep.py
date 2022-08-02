import torch
import torch.nn as nn

import pickle

from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertEmbeddings, BertEncoder, BertPooler

class BertDependencyEmbeddings(BertEmbeddings):
    def __init__(self, config, dep_emb_dim=300, maxlen=512):
        super(BertDependencyEmbeddings, self).__init__(config)
        self.dependency_embeddings = None
        self.fc = nn.Linear(dep_emb_dim, config.hidden_size)
        self.maxlen = maxlen
        
    def create_embeddings(self, path):
        print(f"Loading Pretrained Embedding from {path}")
        state_dict = pickle.load(open(path, "rb"))
        self.dependency_embeddings = nn.Embedding.from_pretrained(state_dict)
        return state_dict

    def add_special_ids(self, unk_id, pad_id):
        self.pad_id = pad_id
        self.skip_ids = set([self.pad_id, unk_id])
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if not self.dependency_embeddings:
            raise Exception("Create the Embeddings First")

        bert_input_ids = input_ids[:, 0, :]
        dep_input_ids = input_ids[:, 1, :]

        bert_seq_length = bert_input_ids.size(1)
        bert_embeddings = self.word_embeddings(bert_input_ids)
        bert_pad_loc = self.maxlen - (bert_input_ids == 0).sum(axis=1)

        # Adding position and token embeddings
        bert_position_ids = torch.arange(bert_seq_length, dtype=torch.long, device=input_ids.device)
        bert_position_ids = bert_position_ids.unsqueeze(0).expand_as(bert_input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(bert_input_ids)

        position_embeddings = self.position_embeddings(bert_position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        bert_embeddings = bert_embeddings + position_embeddings + token_type_embeddings

        dep_embeddings = self.dependency_embeddings(dep_input_ids)
        dep_pad_loc = self.maxlen - (dep_input_ids == 2).sum(axis=1)
        
        # FC for Dependency Embeddings
        fc_dep = self.fc(dep_embeddings.float())
        
        # Combining
        for i, (bert_pad, dep_pad) in enumerate(zip(bert_pad_loc, dep_pad_loc)):
            if bert_pad == self.maxlen:
                continue
                
            left = self.maxlen - bert_pad - dep_pad
            if left <= 0:
                bert_embeddings[i] = torch.concat((bert_embeddings[i, :bert_pad, :], fc_dep[i, :left, :]))[:512]
            else:
                bert_embeddings[i] = torch.concat((bert_embeddings[i, :bert_pad, :], fc_dep[i, :dep_pad, :], bert_embeddings[i, bert_pad:left+bert_pad, :]))
        
        embeddings = self.LayerNorm(bert_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertDependencyModel(BertModel):
    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(BertDependencyModel, self).__init__(config, output_attentions=False, keep_multihead_output=False)
        self.embeddings = BertDependencyEmbeddings(config)
        self.encoder = BertEncoder(
            config,
            output_attentions=output_attentions,
            keep_multihead_output=keep_multihead_output,
        )
        self.pooler = BertPooler(config)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.will_collect_cls_states = False
        self.all_cls_states = []
        self.output_attentions = output_attentions

        self.apply(self.init_bert_weights)

    def create_embeddings(self, path):
        return self.embeddings.create_embeddings(path)

    def add_special_ids(self, unk_id, pad_id):
        self.embeddings.add_special_ids(unk_id, pad_id)
        
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        head_mask=None,
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids[:, 0, :])
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids[:, 0, :])
        embedding_output = self.embeddings(input_ids)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if self.output_attentions:
            output_all_encoded_layers = True
        
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            head_mask=head_mask,
        )
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        pooled_output = self.pooler(encoded_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.output_attentions:
            return all_attentions, logits

        return logits
