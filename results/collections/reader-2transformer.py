from typing import Dict, Iterable, List, Tuple
import json
import torch
import os
from allennlp.data import DatasetReader, Instance, DataLoader, Token
from allennlp.data.fields import LabelField, TextField, ListField, ArrayField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.data import Vocabulary
from allennlp.training.metrics import Metric
from allennlp.training.util import evaluate

from allennlp.data import TextFieldTensors
from allennlp.models import Model 
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.optimizers import AdamWOptimizer, AdamOptimizer, HuggingfaceAdamWOptimizer
from transformers import BertModel, AutoModel
import torch.nn.functional as F
import math
import torch.nn as nn
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.data.samplers import BasicBatchSampler, BucketBatchSampler
# from transformer import Encoder
import numpy as np
from GAT1 import GAT_module
import random
import sys


# In[2]:


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def find_head_idx(source, target):
    source = [token.text for token in source]
    target = [token.text for token in target]
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def to_tuple(sent):
    triple_list = []
    for triple in sent['triple_list']:
        triple_list.append(tuple(triple))
    sent['triple_list'] = triple_list
    
def get_entity_span(entity_heads, entity_tails):
    entities = []
    for head in entity_heads:
        tail = entity_tails[entity_tails > head]
        if len(tail) > 0:
            tail = tail[0]
            entities.append((head, tail))
    return entities


# In[3]:


def seq_gather(seq, entity_span):
    #通过位置获得实体对应向量（求平均）
    vector_list = []
    for i in range(entity_span.shape[0]):
        vectors = []
        for j in range(entity_span.shape[1]):
            h, t = entity_span[i,j][0], entity_span[i,j][1]
            if h==-1 or t==-1:
                vector = torch.zeros(1, seq.shape[-1]).to(device)
            else:
                vector = seq[i][h:t]
                vector = torch.mean(vector, dim=0, keepdim=True)
            vector+=torch.unsqueeze(torch.mean(seq[i]), 0)
#             vector+=seq[i][0:1]
            vectors.append(vector)
        vectors = torch.cat(vectors, dim=0)
        vectors = torch.unsqueeze(vectors, dim=0)
        vector_list.append(vectors)
    gathered_vec = torch.cat(vector_list, dim=0)
    return gathered_vec

def get_triple_list(entity_span, token_ids, rels):
    tokens = [dataset_reader.tokenizer.tokenizer.convert_ids_to_tokens(idx) for idx in token_ids]
    triple_list = []
    for b, rel, sub, obj in rels:
        rel = dataset_reader.id2rel[rel]
        sub_idx = entity_span[b][sub]
        obj_idx = entity_span[b][obj]
        sub_piece = tokens[b][sub_idx[0]:sub_idx[1]]
        obj_piece= tokens[b][obj_idx[0]:obj_idx[1]]
        triple_list.append((piece2word(sub_piece), rel, piece2word(obj_piece)))


    return set(triple_list)

def piece2word(token):
    s = ""
    for i in token:
        if not i.startswith("##"):
            s += ' '
            s += i
        else:
            s += i[2:]
    return s.lstrip()


# In[4]:


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
                triple_token = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
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
            entity_list = sorted(entity_list, key=lambda x:x[0])
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


# In[5]:


class MultiHeadAttentionScore(torch.nn.Module):
    def __init__(self, input_size, output_size, num_heads):
        super(MultiHeadAttentionScore, self).__init__()
        self.num_heads = num_heads
        self.input_size = input_size
        self.output_size = output_size
        self.depth = int(output_size / self.num_heads)
        

        self.GAT = GAT_module(n_layers, num_rels)
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

        return attn_score #BS * NR * NE * NE
    
    


# In[6]:


class CRREL(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 model_name):
        super().__init__(vocab)
#         self.embedder = BertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        self.embedding = PretrainedTransformerEmbedder(model_name).to(device)
        self.embedder = BasicTextFieldEmbedder(token_embedders={'tokens': self.embedding}).to(device)
        self.head_dense = nn.Linear(768, 1).to(device)
        self.tail_dense = nn.Linear(768, 1).to(device)
        self.attn = MultiHeadAttentionScore(768, 768*dataset_reader.num_rels, dataset_reader.num_rels).to(device)
        self.f1 = F1()
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
        input_ids =  text['tokens']['token_ids']
        entity_head_pred = torch.squeeze(self.head_dense(seq_output), dim=-1)
        entity_tail_pred = torch.squeeze(self.tail_dense(seq_output), dim=-1)
        
        entity_head_loss = ((F.binary_cross_entropy_with_logits(entity_head_pred, entity_head, reduction='none')*text_mask).sum(dim=-1) / (text_mask.sum(dim=-1))).mean()
        entity_tail_loss = ((F.binary_cross_entropy_with_logits(entity_tail_pred, entity_tail, reduction='none')*text_mask).sum(dim=-1) / (text_mask.sum(dim=-1))).mean()
        
        entity = seq_gather(seq_output, entity_span)

        rel_pred = self.attn(entity, entity)
        rel_loss = ((F.binary_cross_entropy_with_logits(rel_pred, rel_matrix, reduction='none')*rel_mask).sum(dim=(2,3))/(rel_mask.sum(dim=(2,3)))).mean()
        total_loss = entity_head_loss+entity_tail_loss+rel_loss*alpha
        
        
        #用于dev
        self.f1(entity_head_pred, entity_tail_pred,text_mask,
        seq_output, seq_output, self.attn, rel_matrix, entity_span, input_ids)
        
        return {'loss': total_loss}
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        res = self.f1.get_metric(reset)
        return {"precision": res[0], 'recall':res[1], 'f1_score': res[2]}

        


# In[7]:


from overrides import overrides
class F1(Metric):
    def __init__(self):        
        self.correct_num, self.predict_num, self.gold_num = 1e-10, 1e-10, 1e-10
        
    def __call__(self, entity_head_pred, entity_tail_pred,text_mask,
                sub_seq, obj_seq, attn, rel_matrix, gold_entity_span, input_ids):
        entity_head_score = torch.sigmoid(entity_head_pred)*text_mask
#         print(entity_head_score.shape)
        entity_tail_score = torch.sigmoid(entity_tail_pred)*text_mask
        batch_size = entity_head_score.shape[0]
        entity_heads, entity_tails = torch.where(entity_head_score>threshold), torch.where(entity_tail_score>threshold)
        entity_heads = torch.stack(entity_heads).T
        entity_tails = torch.stack(entity_tails).T
        span = [{'h':[], 't':[]} for _ in range(batch_size)]
        for h in entity_heads:
            span[h[0]]['h'].append(h[1].item())
        for t in entity_tails:
            span[t[0]]['t'].append(t[1].item())  
        get_entity_span(torch.tensor(span[0]['h']), torch.tensor(span[0]['t']))
        spans = [get_entity_span(torch.tensor(s['h']), torch.tensor(s['t'])) for s in span]
        max_num = max([len(s) for s in spans])
        for span in spans:
            while(len(span)<max_num):
                span.append((-1,-1))
        pred_entity_span = torch.tensor(spans)
        if (max_num) > 0:
            sub_pred = seq_gather(sub_seq, pred_entity_span)
            obj_pred = seq_gather(obj_seq, pred_entity_span)
            rel_score = torch.sigmoid(attn(sub_pred, obj_pred))
            rel_pred = torch.stack(torch.where(rel_score>threshold)).T.tolist()

            rel_gold = torch.stack(torch.where(rel_matrix>threshold)).T.tolist()

#             rel_pred = set([tuple(rel) for rel in rel_pred if ((rel[-1]!=rel[-2]) and (pred_entity_span[rel[0]][rel[-1]][0]!=-1) 
#                            and (pred_entity_span[rel[0]][rel[-2]][0]!=-1)
#                                                               )])
            rel_pred = set([tuple(rel) for rel in rel_pred if ((pred_entity_span[rel[0]][rel[-1]][0]!=-1) 
                           and (pred_entity_span[rel[0]][rel[-2]][0]!=-1)
                                                              )])         
            rel_gold = set([tuple(rel) for rel in rel_gold]) 
            pred_triple_list = get_triple_list(pred_entity_span, input_ids, rel_pred)
            gold_tirple_list = get_triple_list(gold_entity_span, input_ids, rel_gold)
#             if pred_triple_list!=gold_tirple_list:
#                 print('pred_triple_list:', pred_triple_list)
#                 print('gold_tirple_list:', gold_tirple_list)
#                 FP = pred_triple_list - (pred_triple_list&gold_tirple_list)
#                 FT = gold_tirple_list - (pred_triple_list&gold_tirple_list)
#                 print('预测错误的：', FP)
#                 print('未被预测正确的:', FT)
#                 global total
#                 total += len([a for a in list(FT) if a[0]==a[-1]])
# #                 print(total)
            
            self.predict_num += len(pred_triple_list)
            self.gold_num += len(gold_tirple_list)
            self.correct_num += len(pred_triple_list&gold_tirple_list)
        else:
            rel_gold = torch.stack(torch.where(rel_matrix>threshold)).T.tolist()
            rel_gold = set([tuple(rel) for rel in rel_gold])
            self.predict_num += 0
            self.gold_num += len(rel_gold)
            self.correct_num += 0
        
    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        precision = self.correct_num / self.predict_num
        recall = self.correct_num / self.gold_num
        f1_score = 2 * precision * recall / (precision + recall)
        
        if reset:
            self.reset()
        return precision, recall, f1_score
    
    @overrides
    def reset(self):
        print("correct_num：{}，gold_num：{}， predict_num：{}".format(self.correct_num, self.gold_num, self.predict_num))
        self.correct_num, self.predict_num, self.gold_num = 1e-10, 1e-10, 1e-10
        
    


# In[8]:


def build_dataset_reader(model_name, rel_dict_path) -> DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(model_name)
    token_indexer = {'tokens': PretrainedTransformerIndexer(model_name)}
    return CR_Reader(tokenizer, token_indexer, 512, rel_dict_path, cache_directory='dataset_cache')
#     return CR_Reader(max_tokens=512, rel_dict_path=rel_dict_path, tokenizer=tokenizer)

def read_data(
    reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = reader.read(train_path)
    validation_data = reader.read(dev_path)
    test_data = reader.read(test_path)
    return training_data, validation_data, test_data

def build_data_loaders(
    train_data: torch.utils.data.Dataset,
    dev_data: torch.utils.data.Dataset,
    test_data : torch.utils.data.Dataset,
    batch_size
) -> Tuple[DataLoader, DataLoader]:
    batch_sampler = BucketBatchSampler(train_data, batch_size=batch_size, sorting_keys=['entity_span'])
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler)
    dev_loader = DataLoader(dev_data, 1, shuffle=False, num_workers=10)
    test_loader = DataLoader(test_data, 1, shuffle=False, num_workers=10)
    return train_loader, dev_loader, test_loader

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader
) -> Trainer:
    parameters = [
        [n, p]
        for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = HuggingfaceAdamWOptimizer(parameters, lr=learning_rate)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        cuda_device=0,
#         patience = patience,
        validation_metric='+f1_score'
    )
    return trainer


# In[9]:


# dataset_name = 'NYT'
dataset_name = sys.argv[2]

train_path = "data/{}/train_triples.json".format(dataset_name)
test_path = "data/{}/test_triples.json".format(dataset_name)
dev_path = "data/{}/dev_triples.json".format(dataset_name)
rel_dict_path = "data/{}/rel2id.json".format(dataset_name)
model_name = 'bert-base-cased'
#model_name ='roberta-base'
# seed = 888
seed = int(sys.argv[1])
#setup_seed(seed)
n_layers = 4
id2rel, _ = json.load(open(rel_dict_path, encoding='utf-8'))
num_rels = len(id2rel)
device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
batch_size = 16
threshold = 0.8
alpha = math.sqrt(num_rels)
# alpha = 10
num_epochs = 200
# patience = 50
learning_rate = 1e-5
serialization_dir = 'bert2trans-l={}-{}-{}--{}--{:.1f}-{}'.format(n_layers,learning_rate, dataset_name, batch_size,alpha,seed)


# In[10]:


dataset_reader = build_dataset_reader(model_name, rel_dict_path)
train_data, dev_data, test_data = read_data(dataset_reader)


vocab = Vocabulary()

train_data.index_with(vocab)
dev_data.index_with(vocab)
test_data.index_with(vocab)

train_loader, dev_loader, test_loader = build_data_loaders(train_data, dev_data,test_data, batch_size)


# In[ ]:


model = CRREL(vocab, model_name)
trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
trainer.train()


# In[ ]:


ckpt = torch.load(os.path.join(serialization_dir, "best.th"))

model.load_state_dict(ckpt)

evaluate(model, test_loader, cuda_device=0)


# In[ ]:


threshold = 0.85
evaluate(model, test_loader, cuda_device=0)


# In[ ]:


threshold = 0.7
evaluate(model, test_loader, cuda_device=0)


# In[ ]:




# In[ ]:




