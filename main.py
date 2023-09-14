import pickle as pkl
from datasets import load_dataset
import numpy as np
from transformers import Trainer
import torch
from tqdm.auto import tqdm
import collections
import evaluate
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForQuestionAnswering

from util import *
from EyeMask import *


model_checkpoint = 'bert-base-cased' # Choose from: 'albert-base-v2', 'bert-base-cased'
feature = 'trt' # Choose from: None, 'trt' (total reading time), 'ffd' (first fixation duration)
task = 'task1' # Choose from: 'task1'(using eye-tracking corpora), 'task2' (using SQuAD benchmark)
mask_type = 'softmax' # Choose from: 'standard', 'exponent', 'softmax'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

train_set, validation_set, train_dataset, validation_dataset, generateMask = getDataset(task, model_checkpoint)

train_dataset_eye = generateMask(train_dataset, train_set, tokenizer, feature, 'train')
validation_dataset_eye = generateMask(validation_dataset, validation_set, tokenizer, feature, 'validation')

if mask_type not in ['standard', 'exp', 'softmax']:
    print('Please set eye-tracking attention mask from \'standard\', \'exponent\', and \'softmax\'!')
else:
    train_eye = combineMask(train_dataset, train_dataset_eye, mask_type)
    validation_eye = combineMask(validation_dataset, validation_dataset_eye, mask_type)

# Fine-tuning
args = TrainingArguments(
    # 'bert_model/corpus_ffd/exp5',
    f'ModelResult/{model_checkpoint}/{task}/{feature}_{mask_type}',
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=2500,
    logging_steps= 1000,
    eval_steps=2000,
    learning_rate=2e-5,
    num_train_epochs=3,
    max_steps=6250,
    weight_decay=0.01,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_eye,
    eval_dataset=validation_eye,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions, _, _ = trainer.predict(validation_eye)
start_logits, end_logits = predictions
compute_metrics(start_logits, end_logits, validation_eye, validation_set)

