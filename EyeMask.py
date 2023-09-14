import pickle as pkl
from datasets import load_dataset
import numpy as np
from transformers import Trainer
import torch
from tqdm.auto import tqdm
import collections
import evaluate
from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoTokenizer, AutoModelForQuestionAnswering
import matplotlib.pyplot as plt


# albert
sp_albert = pkl.load(open('data/ALBERTsp.pkl','rb'))
sp_dict = sp_albert[0]
sp_trial = sp_albert[1]
sp_letter = sp_albert[2]

def buildMaskAlbert(tokens, eye_tracking, eye_text):
    m = 0
    mask = []
    len_word = len(eye_text[0])
    for n in range(len(tokens)):
        if tokens[n] == '[SEP]':
            break
        
        if len(tokens[n]) == 1 and tokens[n].isalpha() == False and tokens[n-1] != '▁':
            mask.append(0)
        else:
            mask.append(eye_tracking[m])
        
        if tokens[n] == '<unk>':
            len_word -= 1
        else:
            len_word -= len(tokens[n].strip('▁') )
        if len_word == 0 and m < (len(eye_text)-1):
            m += 1
            len_word = len(eye_text[m])
        
    return mask

def eyeMaskAlbert(tokenizer, ids, eye_q, eye_t, q_text, eye_text):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    tokens_q = tokens[1:tokens.index('[SEP]')]
    if '<pad>' in tokens:
        tokens_t = tokens[tokens.index('[SEP]')+1:tokens.index('<pad>')-1]
    else:
        tokens_t = tokens[tokens.index('[SEP]')+1:]
    if tokens_t[0][0] != '▁':
        tokens_t[0] = '▁' + tokens_t[0]
    eye_mask = [0]
    
    # question 
    q_mask = buildMaskAlbert(tokens_q, eye_q, q_text)
    eye_mask.extend(q_mask)
    
    eye_mask.append(0)
    
    # text
    if tokens_t[-1] == '[SEP]':
        tokens_t = tokens_t[:-1]
        t_mask = buildMaskAlbert(tokens_t, eye_t, eye_text)
        eye_mask.extend(t_mask)
        eye_mask.append(0)
    else:
        t_mask = buildMaskAlbert(tokens_t, eye_t, eye_text)
        eye_mask.extend(t_mask)
    
    if '<pad>' in tokens:
        eye_mask.append(0)  
    
    eye_mask = np.array(eye_mask)
    eye_mask0 = np.copy(eye_mask)
    
    # padding
    if '<pad>' in tokens:
        padding = np.array(ids[tokens.index('<pad>'):])
        eye_mask = np.concatenate((eye_mask,padding))
    
    return eye_mask

def generateMaskAlbert(dataset, eyeset, tokenizer, feature, set_type=None):
    eye_masks = []
    pointer = 0
    
    if feature == 'trt':
        context_eye = 'contexttrt'
        question_eye = 'questiontrt'
    elif feature == 'ffd':
        context_eye = 'contextffd'
        question_eye = 'questionffd'
    
    for n in tqdm(range(len(dataset))):  
        # tokened trial
        slice_mark = tokenizer.convert_ids_to_tokens(dataset[n]['input_ids']).index('[SEP]')
        text = ''.join(tokenizer.convert_ids_to_tokens(dataset[n]['input_ids'])[slice_mark+1:]).replace('▁',' ').strip('[SEP] ')
        tokend_q = ''.join(tokenizer.convert_ids_to_tokens(dataset[n]['input_ids'])[1:slice_mark]).replace('▁',' ').strip('[SEP] ')

        # extract corresponding eye-tracking trial
        eye_trial = eyeset[pointer]
        context = eye_trial['context'].lower()
        for i in sp_letter:
            context = context.replace(i,sp_letter[i])
        if pointer in sp_trial:
            for i in sp_dict:
                context = context.replace(i,sp_dict[i])
        context = context.replace('’','<unk>').replace('—','<unk>').replace('°','<unk>').replace('₹','<unk>')
        context = context.replace('“','<unk>').replace('”','<unk>').replace('ə','<unk>')

        sub_eye_q = eyeset[pointer]['question'].split()
        
        # pointer moving switch
        moveToNext = False
        continueLast = False
        if n != (len(dataset)-1) and dataset[n]['example_id'] != dataset[n+1]['example_id']:
            moveToNext = True
        if n != 0 and dataset[n]['example_id'] == dataset[n-1]['example_id']:
            continueLast = True  
        
        if (moveToNext == True) and (continueLast == False):
            attention_mask = np.array(dataset[n]['attention_mask'])
            eye_mask = eyeMaskAlbert(tokenizer,dataset[n]['input_ids'], 
                               eye_trial[question_eye], 
                               eye_trial[context_eye], 
                               sub_eye_q,
                               context.replace('<unk>','—').split(), 
                               )
        else:   
            text = text.replace('<pad>','').replace('[SEP]','').strip()
            attention_mask = np.array(dataset[n]['attention_mask'])
            start = context.lower().find(text[:30])
            if moveToNext == True:
                end = -1
            else:
                end = context.find(text[-50:])+49
            start_token = len(eye_trial['context'][:start].split())
            end_token = len(eye_trial['context'][:end].split())

            # uncompleted word
            if tokenizer.convert_ids_to_tokens(dataset[n]['input_ids'])[slice_mark+1][0] != '▁':
                start_token -= 1
            sub_eye_text_trt = eye_trial[context_eye][start_token:end_token]
            sub_eye_text = context.replace('<unk>','—').split()[start_token:end_token]
            
            eye_mask = eyeMaskAlbert(tokenizer,
                               dataset[n]['input_ids'], 
                               eye_trial[question_eye], 
                               sub_eye_text_trt, 
                               sub_eye_q,
                               sub_eye_text, 
                               )
                        
        eye_masks.append(eye_mask)
        if moveToNext == True:
            pointer += 1
    
    return eye_masks

#bert

sp = pkl.load(open('data/BERTsp.pkl','rb'))

# generate eye tracking mask
def buildMaskBert(tokens, eye_tracking, eye_text):
    m = 0
    mask = []
    len_word = len(eye_text[0])
    for n in range(len(tokens)):
        if tokens[n] == '[SEP]':
            break
        if len(tokens[n]) == 1 and tokens[n].isalpha() == False and tokens[n-1] == '#':
            mask.append(0)
        else:
            mask.append(eye_tracking[m])
        if tokens[n] == '[UNK]':
            len_word -= 1
        else:
            len_word -= len(tokens[n].strip('#') )
            
        if len_word == 0 and m < (len(eye_text)-1):
            m += 1
            len_word = len(eye_text[m])
                
    return mask

def eyeMaskBert(tokenizer, ids, eye_q, eye_t, q_text, eye_text):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    tokens_q = tokens[1:tokens.index('[SEP]')]
    if '[PAD]' in tokens:
        tokens_t = tokens[tokens.index('[SEP]')+1:tokens.index('[PAD]')-1]
    else:
        tokens_t = tokens[tokens.index('[SEP]')+1:]
    eye_mask = [0]
    
    # question 
    q_mask = buildMaskBert(tokens_q, eye_q, q_text)
    eye_mask.extend(q_mask)
    
    eye_mask.append(0)
    
    # text
    if tokens_t[-1] == '[SEP]':
        tokens_t = tokens_t[:-1]
        t_mask = buildMaskBert(tokens_t, eye_t, eye_text)
        eye_mask.extend(t_mask)
        eye_mask.append(0)
    else:
        t_mask = buildMaskBert(tokens_t, eye_t, eye_text)
        eye_mask.extend(t_mask)
    
    if '[PAD]' in tokens:
        eye_mask.append(0)  
    
    eye_mask = np.array(eye_mask)
    eye_mask0 = np.copy(eye_mask)
    
    # padding
    if '[PAD]' in tokens:
        padding = np.array(ids[tokens.index('[PAD]'):])
        eye_mask = np.concatenate((eye_mask,padding))
    
    return eye_mask

def generateMaskBert(dataset, eyeset, tokenizer, feature, set_type):
    eye_masks = []
    pointer = 0
    
    if feature == 'trt':
        context_eye = 'contexttrt'
        question_eye = 'questiontrt'
    elif feature == 'ffd':
        context_eye = 'contextffd'
        question_eye = 'questionffd'
    
    dict_sp = sp[set_type]
    
    for n in tqdm(range(len(dataset))):            
        # tokened trial
        slice_mark = tokenizer.convert_ids_to_tokens(dataset[n]['input_ids']).index('[SEP]')
        text = ' '.join(tokenizer.convert_ids_to_tokens(dataset[n]['input_ids'])[slice_mark+1:]).replace(' #','#').replace('#','').strip('[SEP] ')
        tokend_q = ' '.join(tokenizer.convert_ids_to_tokens(dataset[n]['input_ids'])[1:slice_mark]).replace(' #','#').replace('#','').strip('[SEP] ')

        # extract corresponding eye-tracking trial
        eye_trial = eyeset[pointer]
        context = eye_trial['context'].replace('\n',' ')
        
        if pointer in dict_sp:
            for i in dict_sp[pointer]:
                context = context.replace(i,dict_sp[pointer][i])
            
        context = context.lower()
            
        sub_eye_q = eyeset[pointer]['question'].split()
        
        # pointer moving switch
        continueLast = False
        moveToNext = False
        
        # whether next trial is a new question or the answer is already found before scanning full text
        if n != (len(dataset)-1) and dataset[n]['input_ids'][:slice_mark] != dataset[n+1]['input_ids'][:slice_mark]:
            # if the question is already found before move to the truncated part
            moveToNext = True
        elif dataset[n]['input_ids'][-1] == 0: # same question for different texts
            moveToNext = True
        
        if n != 0 and dataset[n]['input_ids'][:slice_mark] == dataset[n-1]['input_ids'][:slice_mark]:
            continueLast = True

        if (moveToNext == True) and (continueLast == False):
            attention_mask = np.array(dataset[n]['attention_mask'])
            eye_mask = eyeMaskBert(tokenizer,dataset[n]['input_ids'], 
                               eye_trial[question_eye], 
                               eye_trial[context_eye], 
                               sub_eye_q,
                               context.split(), 
                               )
        else:   
            text = text.replace('[PAD]','').replace('[SEP]','').strip().lower()
            text = text.replace(' ,',',').replace(' \'','\'').replace(' .','.').replace(' – ','–').replace(' - ','-').replace('- ','-')
            text = text.replace('\' s','\'s').replace('0. ','0.').replace(' %','%').replace(' ?','?').replace(' ;',';').replace(' :',':')
            text = text.replace('. 0%','.0%').replace('. 1%','.1%').replace('. 2%','.2%').replace('. 3%','.3%').replace('. 4%','.4%').replace('. 5%','.5%').replace('. 6%','.6%').replace('. 7%','.7%').replace('. 8%','.8%').replace('. 9%','.9%')
            text = text.replace('( ','(').replace(' )', ')').replace('[ ','[').replace(' ]',']')
            #special cases
            text = text.replace(', 600',',600').replace('1. 8','1.8').replace('.f','. f').replace('. ob)','.ob)').replace('4. 8','4.8')
            text = text.replace('3–5', '3 – 5').replace('3, 498','3,498').replace(', 552',',552').replace('1. 6','1.6').replace('x. 25','x.25')
            text = text.replace('u. s.','u.s.').replace('03: 50', '03:50').replace('s - i', 's-i').replace('" demos "','"demos"')
            attention_mask = np.array(dataset[n]['attention_mask'])
            if continueLast == False:
                start = 0
            else:
                start = context.replace('"','').find(text.replace('"','').replace('  ',' ')[:25])
                start = start + context[:start].count('"')
            if moveToNext == True:
                end = -1
            else:
                end = context.replace('"','').find(text.replace('"','').replace('  ',' ')[-40:])+39
                end = end + context[:end].count('"')
            start_token = len(eye_trial['context'][:start].split())
            end_token = len(eye_trial['context'][:end].split())
                
            # uncompleted word
            if tokenizer.convert_ids_to_tokens(dataset[n]['input_ids'])[slice_mark+1][0] == '#':
                first_word = tokenizer.convert_ids_to_tokens(dataset[n]['input_ids'])[slice_mark+1][0].replace('#','')
                start_token -= 1
            sub_eye_text_trt = eye_trial[context_eye][start_token:end_token]
            sub_eye_text = context.split()[start_token:end_token]
            if tokenizer.convert_ids_to_tokens(dataset[n]['input_ids'])[slice_mark+1][0] == '#':
                sub_eye_text[0] = sub_eye_text[0][sub_eye_text[0].find(first_word):]
            
            eye_mask = eyeMaskBert(tokenizer,
                               dataset[n]['input_ids'], 
                               eye_trial[question_eye], 
                               sub_eye_text_trt, 
                               sub_eye_q,
                               sub_eye_text, 
                               )
                        
        eye_masks.append(eye_mask)
        if moveToNext == True:
            pointer += 1
    
    return eye_masks


# combine

def combineMask(dataset, raw_eye_masks, generateMethod=None):
    
    eye_masks = []
    for n in tqdm(range(len(raw_eye_masks))):
        attention_mask = np.array(dataset[n]['attention_mask'])
        eye_mask = raw_eye_masks[n]
        if generateMethod == 'standard':
            eye_mask = eye_mask/np.max(eye_mask) / 10000
        else:
            eye_mask = np.exp(eye_mask - np.max(eye_mask))
            if generateMethod == 'exp':
                eye_mask = eye_mask / 10000
            else:
                eye_mask = eye_mask / eye_mask.sum(axis=0)
                if generateMethod == 'softmax':
                    eye_mask = eye_mask / 10000
                if generateMethod == 'reverse':
                    eye_mask = eye_mask0 - 1000 * (1 - eye_mask)
                    eye_mask = np.exp(eye_mask - np.max(eye_mask))
                    eye_mask = (eye_mask / eye_mask.sum(axis=0)) / 100
        eye_mask = attention_mask + eye_mask# - np.max(eye_mask)
        eye_masks.append(eye_mask.tolist())
                
    dataset = dataset.remove_columns("attention_mask")
    dataset = dataset.add_column("attention_mask", eye_masks)
    
    return dataset



# visualize

def visual(sent, 
           model, 
           tokenizer, 
           print_item, 
           model_type, 
           eye_type, 
           mask_type,
           data,
           which=None,
           save=False):
    
    model = model.to('cpu')

    inputs = tokenizer(sent, return_tensors='pt').to('cpu')
    outputs = model(**inputs, output_attentions=True)
    print(len(outputs['attentions']))
    
    def make_image(data):
        fig = plt.imshow(data)
    #     fig.set_cmap('hot')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    
    fig, axs = plt.subplots(3, 4, figsize=(20, 20))
            
    if print_item == 'head':
        head = outputs['attentions'][which-1][0].detach().numpy()
        for n in range(12):
            ax = plt.subplot(3,4,n+1)
            plt.subplot(3,4,n+1)
            make_image(head[n])
            ax.set_title(f'head{n+1}', fontsize=18)
    
    if print_item == 'layer':
        layer = []
        for n in range(12):
            layer.append(outputs['attentions'][n][0][which-1].detach().numpy())
        for n in range(12):
            ax = plt.subplot(3,4,n+1)
            plt.subplots_adjust(bottom=0.1, left=0.1, right=0.7, top=0.55)
            plt.subplot(3,4,n+1)
            make_image(layer[n])
            ax.set_title(f'layer{n+1}', fontsize=18)
    if save:    
        plt.savefig(f'{model_type}{data}-{eye_type}-{mask_type}-head{which}.png',dpi=600)
    plt.show()
    