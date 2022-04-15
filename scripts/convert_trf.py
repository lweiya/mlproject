
model_checkpoint = "bert-base-chinese"

from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

example = {
    'text':"\n这意味着我们需要对我们的标签做\n一些处理，因为标记化器返回的输入id比我们的数据集所包含的标签列表要长，首先是因为一些特殊的标记可能被添加（我们可以一个[CLS]和一个[SEP]以上），然后是因为那些可能被分割成多个标记的词。",
    'lable':[[5,6,"主语"],[9,11,"谓语"]]
}

tokenized_input = tokenizer(example["text"])
len(example['text'])
len(tokenized_input['input_ids'])

example['text'][4:6]