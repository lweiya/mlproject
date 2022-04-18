
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

# b_trans_dataset_bio(bio_labels,'train.json')
# b_trans_dataset_bio(bio_labels,'dev.json')

datasets = load_dataset('json', data_files= {'train': ASSETS_PATH + 'train_trf.json', 'dev': ASSETS_PATH + 'dev_trf.json'})

label_all_tokens = True
# 对齐label
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["data"], truncation=True, is_split_into_words=True)

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
tokenized_datasets.remove_columns(["data", "label"])

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

data_collator = DataCollatorForTokenClassification(tokenizer,padding='max_length',max_length=512)

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








