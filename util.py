import collections
import numpy as np
from tqdm.auto import tqdm
import torch
import pickle as pkl
from transformers import AutoModelForQuestionAnswering
import evaluate
from transformers import AutoTokenizer, TrainingArguments, AutoTokenizer, AutoModelForQuestionAnswering
from EyeMask import *

# preprocessing
def getDataset(task, model_checkpoint):
    max_length = 384
    stride = 128
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if task == 'task1':
        train_set = pkl.load(open('data/train_eye.pkl', 'rb'))
        validation_set = pkl.load(open('data/val_eye.pkl', 'rb'))
    elif task == 'task2':
        train_set = pkl.load(open('data/squadTrain.pkl', 'rb'))
        validation_set = pkl.load(open('data/squadVal.pkl', 'rb'))
    else:
        print('Please choose from \'task1\' and \'task2\'!')

    train_dataset = generate_train(train_set, tokenizer, max_length, stride)
    validation_dataset = generate_validation(validation_set, tokenizer, max_length, stride)

    if model_checkpoint == 'bert-base-cased':
        generateMask = generateMaskBert
    elif model_checkpoint == 'albert-base-v2':
        generateMask = generateMaskAlbert
    else:
        print('Please choose from \'bert-base-cased\' and \'albert-base-v2\'!')

    return train_set, validation_set, train_dataset, validation_dataset, generateMask

def generate_train(train_set, tokenizer, max_length, stride):
    
    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        titles = examples['id']
        titlelist = []
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            title = titles[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
            
            titlelist.append(title)

        inputs["example_id"] = titlelist
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    train_dataset = train_set.map(
                    preprocess_training_examples,
                    batched=True,
                    remove_columns=train_set.column_names,
                )
    return train_dataset

def generate_validation(validation_set, tokenizer, max_length, stride):
    
    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]
            
        inputs["example_id"] = example_ids
        return inputs

    validation_dataset = validation_set.map(
                            preprocess_validation_examples,
                            batched=True,
                            remove_columns=validation_set.column_names,
                        )
    return validation_dataset

## Post processing

def postProcessing(validation_set,tokenizer, model_checkpoint, max_length, stride):
    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    
    
    small_eval_set = validation_set.select(range(100))
    trained_checkpoint = "distilbert-base-cased-distilled-squad"

    tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
    eval_set = small_eval_set.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=validation_set.column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
    eval_set_for_model.set_format("torch")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
    trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(
        device
    )

    with torch.no_grad():
        outputs = trained_model(**batch)

    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()


    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(eval_set):
        example_to_features[feature["example_id"]].append(idx)


    n_best = 20
    max_answer_length = 30
    predicted_answers = []

    for example in small_eval_set:
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = eval_set["offset_mapping"][feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answers.append(
                        {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                    )

        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

    metric = evaluate.load("squad")

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
    ]
#     print(metric.compute(predictions=predicted_answers, references=theoretical_answers))
    return start_logits, end_logits

n_best = 20
max_answer_length = 30
predicted_answers = []
metric = evaluate.load("squad")

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    predicted_logits = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        'logits':(offsets[start_index][0], offsets[end_index][1]),
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)
        
        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
            predicted_logits.append({"id": example_id, "logits": best_answer["logits"]})
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    
    theoretical_logits = [{"id": ex["id"], "logits": ex["answers"]} for ex in examples]
    
    logits = []
    dict_eval = {'precision':[],'IoU':[],'sufficiency':[],'comprehensiveness':[]}
    for m in range(len(predicted_logits)):
        trial = {'id':predicted_logits[m]['id']}
        predicted = []
        theoretical = []
        for n in range(predicted_logits[m]['logits'][0],predicted_logits[m]['logits'][1]):
            predicted.append(n)
        a = theoretical_answers[m]['answers']['answer_start'][0]
        for n in range(len(theoretical_answers[m]['answers']['text'][0])):
            theoretical.append(a)
            a += 1
        predicted = set(predicted)
        theoretical = set(theoretical)
        logits.append([predicted, theoretical])
        
        dict_eval['precision'].append(len(predicted & theoretical)/len(predicted))
        dict_eval['IoU'].append(len(predicted & theoretical)/(len(predicted)+len(theoretical)-len(predicted & theoretical)))
        dict_eval['sufficiency'].append(len(predicted & theoretical)/len(theoretical))
        dict_eval['comprehensiveness'].append(int(theoretical.issubset(predicted)))
    
#     dict_eval['auprc'] = roc_auc_score(dict_eval['sufficiency'], dict_eval['precision'])
    dict_eval['precision'] = np.mean(np.array(dict_eval['precision']))*100
    dict_eval['IoU'] = np.mean(np.array(dict_eval['IoU']))*100
    dict_eval['sufficiency'] = np.mean(np.array(dict_eval['sufficiency']))*100
    dict_eval['comprehensiveness'] = np.sum(np.array(dict_eval['comprehensiveness']))/len(dict_eval['comprehensiveness'])*100
    
    
    subdict = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    dict_eval['accuracy'] = subdict['exact_match']
    dict_eval['f1_score'] = subdict['f1']
    return dict_eval


