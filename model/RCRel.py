from typing import Dict, Iterable, List, Tuple
import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from utils import *
import torch.nn.functional as F
import math
import torch.nn as nn
from model.GAT import GAT_module
from Metrics.F1 import F1
device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")


class CRREL(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 model_name,
                 dataset_reader,
                 alpha,
                 n_layers,
                 dataset_name,
                 threshold
                 ):
        super().__init__(vocab)
        #         self.embedder = BertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        self.embedding = PretrainedTransformerEmbedder(model_name).to(device)
        self.embedder = BasicTextFieldEmbedder(token_embedders={'tokens': self.embedding}).to(device)
        self.head_dense = nn.Linear(768, 1).to(device)
        self.tail_dense = nn.Linear(768, 1).to(device)
        self.attn = MultiHeadAttentionScore(768, 768 * dataset_reader.num_rels, dataset_reader.num_rels, n_layers, dataset_reader.num_rels, dataset_name).to(device)
        self.f1 = F1(threshold=threshold, dataset_reader=dataset_reader)
        self.alpha = alpha


    def forward(self,
                text: Dict[str, torch.Tensor],
                text_mask: torch.Tensor,
                entity_span: List,
                entity_head: torch.Tensor,
                entity_tail: torch.Tensor,
                rel_matrix: torch.Tensor,
                rel_mask: torch.Tensor
                ):
        seq_output = self.embedder(text)
        input_ids = text['tokens']['token_ids']
        entity_head_pred = torch.squeeze(self.head_dense(seq_output), dim=-1)
        entity_tail_pred = torch.squeeze(self.tail_dense(seq_output), dim=-1)

        entity_head_loss = ((F.binary_cross_entropy_with_logits(entity_head_pred, entity_head,
                                                                reduction='none') * text_mask).sum(dim=-1) / (
                                text_mask.sum(dim=-1))).mean()
        entity_tail_loss = ((F.binary_cross_entropy_with_logits(entity_tail_pred, entity_tail,
                                                                reduction='none') * text_mask).sum(dim=-1) / (
                                text_mask.sum(dim=-1))).mean()

        entity = seq_gather(seq_output, entity_span)

        rel_pred = self.attn(entity, entity)
        rel_loss = ((F.binary_cross_entropy_with_logits(rel_pred, rel_matrix, reduction='none') * rel_mask).sum(
            dim=(2, 3)) / (rel_mask.sum(dim=(2, 3)))).mean()
        total_loss = entity_head_loss + entity_tail_loss + rel_loss * self.alpha

        # 用于dev
        self.f1(entity_head_pred, entity_tail_pred, text_mask,
                seq_output, seq_output, self.attn, rel_matrix, entity_span, input_ids)

        return {'loss': total_loss}

    def get_metrics(self, reset: bool = False):
        res = self.f1.get_metric(reset)
        return {"precision": res[0], 'recall': res[1], 'f1_score': res[2]}


class MultiHeadAttentionScore(torch.nn.Module):
    def __init__(self, input_size, output_size, num_heads, n_layers, num_rels, dataset_name):
        super(MultiHeadAttentionScore, self).__init__()
        self.num_heads = num_heads
        self.input_size = input_size
        self.output_size = output_size
        self.depth = int(output_size / self.num_heads)

        self.GAT = GAT_module(n_layers, num_rels, dataset_name)
        self.linear = torch.nn.Linear(input_size, output_size)
        self.linear1 = torch.nn.Linear(output_size, input_size)

        self.Wq = torch.nn.Linear(input_size, output_size)
        self.Wk = torch.nn.Linear(input_size, output_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)  # BS * SL * NH * H
        return x.permute([0, 2, 1, 3])  # BS * NH * SL * H

    def forward(self, x, _):  # BS * NE * HS
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = self.linear(x)
        x = self.GAT(x).reshape([batch_size, seq_len, self.output_size])
        x = self.linear1(x)

        q = self.Wq(x)  # BS * SL * OUT
        k = self.Wk(x)  # BS * SL * OUT

        q = self.split_into_heads(q, batch_size)  # BS * NH * SL * H
        k = self.split_into_heads(k, batch_size)  # BS * NH * SL * H

        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2))
        attn_score = attn_score / math.sqrt(k.shape[-1])

        return attn_score  # BS * NR * NE * NE