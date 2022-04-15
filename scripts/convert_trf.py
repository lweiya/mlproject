from transformers import AutoTokenizer
import json

ROOT_PATH = '../'
ASSETS_PATH = ROOT_PATH + 'assets/'

# 读取labels.txt
def read_labels(file):
    labels = []
    with open(file, 'r',encoding='utf8') as f:
        for line in f:
            labels.append(line.strip())
    return labels

labels = read_labels('labels.txt')

# labels每个项目后面增加b
label_b = [ label + 'b' for label in labels]
label_i = [ label + 'i' for label in labels]

labels = []
for lab,lai in zip(label_b,label_i):
    labels.append(lab)
    labels.append(lai)

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

example = train[0]


model = "bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model)

label_all_tokens = True

tokenized_inputs = tokenizer(example["data"])
o_labels = example['label']
test_label = o_labels[0]
label_ = test_label[2]
label_b = label_ + 'b'
lable_i = label_ + 'i'
# 查找label_b 在 labels中的位置
label_b_index = labels.index(label_b) + 1
label_i_index = labels.index(lable_i) + 1
start = test_label[0]
end = test_label[1]

words_ids = tokenized_inputs.word_ids()
# 生成label_ids 
label_ids = len(words_ids) * [0]
idx = words_ids.index(None)
start_idx = words_ids.index(start)
end_idx = words_ids.index(end)
label_ids[idx] = -100
label_ids[start_idx] = label_b_index
label_ids[(start_idx + 1):end_idx] = [label_i_index] * (end-start)

for i in label_ids:
    print(i)



for i, label in enumerate(example["label"]):
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
