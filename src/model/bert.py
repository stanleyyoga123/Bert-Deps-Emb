import torch
import torch.nn as nn

import pickle

from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertEmbeddings, BertEncoder, BertPooler


class BertClassifier(BertModel):
    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(BertClassifier, self).__init__(config, output_attentions=False, keep_multihead_output=False)
        self.embeddings = BertEmbeddings(config)
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
        embedding_output = self.embeddings(input_ids[:, 0, :])
        
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
        