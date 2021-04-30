from typing import Dict, Iterable, List, Tuple
import json
import torch
from allennlp.data import DatasetReader, Instance, DataLoader, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training.optimizers import AdamWOptimizer, AdamOptimizer, HuggingfaceAdamWOptimizer
import math
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.data.samplers import BasicBatchSampler, BucketBatchSampler
from DataLoader.dataReader import CR_Reader
from model.RCRel import CRREL
import sys



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

if __name__ == '__main__':

    dataset_name = sys.argv[1]
    seed = int(sys.argv[2])
    train_path = "data/{}/train_triples.json".format(dataset_name)
    test_path = "data/{}/test_triples.json".format(dataset_name)
    dev_path = "data/{}/dev_triples.json".format(dataset_name)
    rel_dict_path = "data/{}/rel2id.json".format(dataset_name)
    model_name = 'bert-base-cased'

    #setup_seed(seed)
    n_layers = 4
    id2rel, _ = json.load(open(rel_dict_path, encoding='utf-8'))
    num_rels = len(id2rel)
    device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
    batch_size = 16
    threshold = 0.8
    alpha = math.sqrt(num_rels)
    num_epochs = 200
    learning_rate = 1e-5
    serialization_dir = './results/bert2trans-l={}-{}-{}--{}--{:.1f}-{}'.format(n_layers,learning_rate, dataset_name, batch_size,alpha,seed)

    dataset_reader = build_dataset_reader(model_name, rel_dict_path)
    train_data, dev_data, test_data = read_data(dataset_reader)


    vocab = Vocabulary()
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    test_data.index_with(vocab)

    train_loader, dev_loader, test_loader = build_data_loaders(train_data, dev_data,test_data, batch_size)
    model = CRREL(vocab, model_name, dataset_reader, alpha, n_layers, dataset_name,threshold)
    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
    trainer.train()