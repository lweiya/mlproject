from transformers import AutoTokenizer
import json

ROOT_PATH = '../'
ASSETS_PATH = ROOT_PATH + 'assets/'

# 读取文本文件转换成为json到list当中
def b_read_dataset(file):
    data = d_read_json(ASSETS_PATH + file)
    return data

# 读取数据集文件返回list
def d_read_json(path) -> list:
    data = []
    with open(path, 'r',encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train = b_read_dataset('train.json')

examples = train[0]

model = "bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model)

text = "你好，我在北京天安门"

l_t = list(text)

tokenizer(text)

tokens = tokenizer(text)

tokenizer.convert_ids_to_tokens(tokens['input_ids'])

tokens.word_ids()

word_ids = tokenized_input.word_ids()
aligned_labels = [-100 if i is None else example[f"{task}_tags"][i] for i in word_ids]

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

label_all_tokens = True

tokenized_inputs = tokenizer(examples["data"])
labels = []
for i, label in enumerate(examples["label"]):
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        # Special tokens have a word id that is None. We set the label to -100 so they are automatically
        # ignored in the loss function.
        if word_idx is None:
            label_ids.append(-100)
        # We set the label for the first token of each word.
        elif word_idx != previous_word_idx:
            label_ids.append(label[word_idx])
        # For the other tokens in a word, we set the label to either the current label or -100, depending on
        # the label_all_tokens flag.
        else:
            label_ids.append(label[word_idx] if label_all_tokens else -100)
        previous_word_idx = word_idx

    labels.append(label_ids)
tokenized_inputs["labels"] = labels

