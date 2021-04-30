from typing import Dict, Iterable, List, Tuple
import json
import torch
import os
from allennlp.data import DatasetReader, Instance, DataLoader, Token
from allennlp.data.fields import LabelField, TextField, ListField, ArrayField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from utils import *


class CR_Reader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 rel_dict_path: str = None,
                 **kwargs
                 ):
        super().__init__()
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.id2rel, self.rel2id = json.load(open(rel_dict_path, encoding='utf-8'))
        self.id2rel = {int(i): j for i, j in self.id2rel.items()}
        self.num_rels = len(self.id2rel)

    def _read(self, file_path: str) -> Iterable[Instance]:
        data = json.load(open(file_path, encoding='utf-8'))
        for sent in data:
            to_tuple(sent)
        for sent in data:
            text = sent['text']
            triple_list = sent["triple_list"]
            tokens = self.tokenizer.tokenize(text)
            if self.max_tokens:
                tokens = tokens[:self.max_tokens]
            text_len = len(tokens)
            text_mask = torch.ones(text_len)
            entity_list = []
            triple_idx_list = []
            # 获取所有实体的位置
            for triple in triple_list:
                triple_token = (
                self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                sub = triple_token[0]
                rel = triple_token[1]
                obj = triple_token[2]
                sub_head_idx = find_head_idx(tokens, sub)
                obj_head_idx = find_head_idx(tokens, obj)
                #                 if sub != obj:
                #                     obj_head_idx = find_head_idx(tokens, obj)
                #                 else:
                #                     obj_head_idx = find_head_idx(tokens[sub_head_idx+1:], obj)
                #                     if obj_head_idx == -1:
                #                         obj_head_idx = sub_head_idx

                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(sub))
                    obj = (obj_head_idx, obj_head_idx + len(obj))
                    if sub not in entity_list:
                        entity_list.append(sub)
                    if obj not in entity_list:
                        entity_list.append(obj)
                    triple_idx_list.append([sub, obj, self.rel2id[rel]])
                else:
                    raise ValueError
            #             if len(entity_list) == 1:
            #                 entity_list+=entity_list
            num_entity = len(entity_list)
            entity_list = sorted(entity_list, key=lambda x: x[0])
            # 标注实体的头尾位置
            entity_head = torch.zeros(text_len)
            entity_tail = torch.zeros(text_len)
            for h_idx, t_idx in entity_list:
                entity_head[h_idx] = 1
                entity_tail[t_idx] = 1

            rel_matrix = torch.zeros([self.num_rels, num_entity, num_entity])

            for triple_idx in triple_idx_list:
                rel_matrix[triple_idx[2], entity_list.index(triple_idx[0]), entity_list.index(triple_idx[1])] = 1
            #             rel_eye = torch.unsqueeze(torch.eye(num_entity), dim=0).repeat_interleave(self.num_rels, dim=0)
            rel_mask = torch.ones([self.num_rels, num_entity, num_entity])

            text_field = TextField(tokens, self.token_indexers)
            text_mask_field = ArrayField(text_mask)
            entity_span_field = ListField([SpanField(start, end, text_field) for start, end in entity_list])
            entity_head_field = ArrayField(entity_head)
            entity_tail_field = ArrayField(entity_tail)
            rel_mask_field = ArrayField(rel_mask)
            rel_field = ArrayField(rel_matrix)

            fields = {'text': text_field,
                      'text_mask': text_mask_field,
                      'entity_span': entity_span_field,
                      'entity_head': entity_head_field,
                      'entity_tail': entity_tail_field,
                      'rel_matrix': rel_field,
                      'rel_mask': rel_mask_field
                      }
            yield Instance(fields)