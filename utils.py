import torch
device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")


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
    # 通过位置获得实体对应向量（求平均）
    vector_list = []
    for i in range(entity_span.shape[0]):
        vectors = []
        for j in range(entity_span.shape[1]):
            h, t = entity_span[i, j][0], entity_span[i, j][1]
            if h == -1 or t == -1:
                vector = torch.zeros(1, seq.shape[-1]).to(device)
            else:
                vector = seq[i][h:t]
                vector = torch.mean(vector, dim=0, keepdim=True)
            vector += torch.unsqueeze(torch.mean(seq[i]), 0)
            #             vector+=seq[i][0:1]
            vectors.append(vector)
        vectors = torch.cat(vectors, dim=0)
        vectors = torch.unsqueeze(vectors, dim=0)
        vector_list.append(vectors)
    gathered_vec = torch.cat(vector_list, dim=0)
    return gathered_vec


def get_triple_list(entity_span, token_ids, rels, dataset_reader):
    tokens = [dataset_reader.tokenizer.tokenizer.convert_ids_to_tokens(idx) for idx in token_ids]
    triple_list = []
    for b, rel, sub, obj in rels:
        rel = dataset_reader.id2rel[rel]
        sub_idx = entity_span[b][sub]
        obj_idx = entity_span[b][obj]
        sub_piece = tokens[b][sub_idx[0]:sub_idx[1]]
        obj_piece = tokens[b][obj_idx[0]:obj_idx[1]]
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