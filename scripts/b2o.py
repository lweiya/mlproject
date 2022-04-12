from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl

path = '../assets/test.json'

dataset = read_jsonl(filepath=path, dataset=NERDataset, encoding='utf-8')
dataset.to_conll2003(tokenizer=str.split)
dataset.to_spacy(tokenizer=str.split)

# 保存dataset
dataset.save(path='../assets/dataset')
