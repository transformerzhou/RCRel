from overrides import overrides
from typing import Dict, Iterable, List, Tuple
import torch
from allennlp.training.metrics import Metric
from utils import *



class F1(Metric):
    def __init__(self, threshold, dataset_reader):
        self.correct_num, self.predict_num, self.gold_num = 1e-10, 1e-10, 1e-10
        self.threshold = threshold
        self.dataset_reader = dataset_reader

    def __call__(self, entity_head_pred, entity_tail_pred, text_mask,
                 sub_seq, obj_seq, attn, rel_matrix, gold_entity_span, input_ids):
        entity_head_score = torch.sigmoid(entity_head_pred) * text_mask
        #         print(entity_head_score.shape)
        entity_tail_score = torch.sigmoid(entity_tail_pred) * text_mask
        batch_size = entity_head_score.shape[0]
        entity_heads, entity_tails = torch.where(entity_head_score > self.threshold), torch.where(
            entity_tail_score > self.threshold)
        entity_heads = torch.stack(entity_heads).T
        entity_tails = torch.stack(entity_tails).T
        span = [{'h': [], 't': []} for _ in range(batch_size)]
        for h in entity_heads:
            span[h[0]]['h'].append(h[1].item())
        for t in entity_tails:
            span[t[0]]['t'].append(t[1].item())
        get_entity_span(torch.tensor(span[0]['h']), torch.tensor(span[0]['t']))
        spans = [get_entity_span(torch.tensor(s['h']), torch.tensor(s['t'])) for s in span]
        max_num = max([len(s) for s in spans])
        for span in spans:
            while (len(span) < max_num):
                span.append((-1, -1))
        pred_entity_span = torch.tensor(spans)
        if (max_num) > 0:
            sub_pred = seq_gather(sub_seq, pred_entity_span)
            obj_pred = seq_gather(obj_seq, pred_entity_span)
            rel_score = torch.sigmoid(attn(sub_pred, obj_pred))
            rel_pred = torch.stack(torch.where(rel_score > self.threshold)).T.tolist()

            rel_gold = torch.stack(torch.where(rel_matrix > self.threshold)).T.tolist()

            #             rel_pred = set([tuple(rel) for rel in rel_pred if ((rel[-1]!=rel[-2]) and (pred_entity_span[rel[0]][rel[-1]][0]!=-1)
            #                            and (pred_entity_span[rel[0]][rel[-2]][0]!=-1)
            #                                                               )])
            rel_pred = set([tuple(rel) for rel in rel_pred if ((pred_entity_span[rel[0]][rel[-1]][0] != -1)
                                                               and (pred_entity_span[rel[0]][rel[-2]][0] != -1)
                                                               )])
            rel_gold = set([tuple(rel) for rel in rel_gold])
            pred_triple_list = get_triple_list(pred_entity_span, input_ids, rel_pred, self.dataset_reader)
            gold_tirple_list = get_triple_list(gold_entity_span, input_ids, rel_gold, self.dataset_reader)
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
            self.correct_num += len(pred_triple_list & gold_tirple_list)
        else:
            rel_gold = torch.stack(torch.where(rel_matrix > self.threshold)).T.tolist()
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