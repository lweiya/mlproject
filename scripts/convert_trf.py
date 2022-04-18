
from transformers import AutoTokenizer
from data_utils import *
import numpy as np
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

model_checkpoint = "bert-base-chinese"
task = 'ner'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

bio_labels = b_generate_biolabels_from('labels.txt')

# max_length = 510

# data = b_read_dataset('dev_trf.json')

# new_data = []
# for sample in data:
#     data_text = sample["data"]
#     data_label = sample["label"]

#     divs = len(data_text)/max_length + 1

#     every = len(data_text)// divs

#     befor_after = (max_length - every ) // 2


#     for i in range (0,int(divs)):
#         new_sample = {}
#         start  = i * every
#         end = (i+1) * every
#         if i == 0:
#             end = end + befor_after * 2
#         elif i == int(divs) - 1:
#             start = start - befor_after * 2
#         else:
#             start = start - befor_after
#             end = end + befor_after
#         start = start if start >= 0 else 0
#         end = end if end <= len(data_text) else len(data_text)
#         start = int(start)
#         end = int(end)
#         new_text_data = data_text[start:end]
#         new_label_data = data_label[start:end]
#         new_sample["data"] = new_text_data
#         new_sample["label"] = new_label_data
#         new_data.append(new_sample)

# for sample in new_data:
#     print(len(sample["data"]),len(sample["label"]))

# b_save_list_datasets(new_data,'train_trf_2.json')
# b_save_list_datasets(new_data,'dev_trf_2.json')

# b_trans_dataset_bio(bio_labels,'train.json')
# b_trans_dataset_bio(bio_labels,'dev.json')

datasets = load_dataset('json', data_files= {'train': ASSETS_PATH + 'train_trf_2.json', 'dev': ASSETS_PATH + 'dev_trf_2.json'})

label_all_tokens = True
# 对齐label
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["data"], truncation=True, is_split_into_words=True,padding='max_length', max_length=512)

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
    return tokenized_inputs

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
# 去掉 data,label
tokenized_datasets = tokenized_datasets.remove_columns(["data", "label"])

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(bio_labels))

model_name = model_checkpoint.split("/")[-1]

batch_size = 16

args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=50,
    weight_decay=0.01,
)


metric = load_metric("seqeval")

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [bio_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [bio_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['dev'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()








