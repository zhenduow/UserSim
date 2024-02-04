import csv
import json 
import pandas as pd
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import torch as T
import os
from tqdm import tqdm
import re
from autocorrect import Speller
spell = Speller(lang='en')

import torch as T
import pandas as pd
import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


# USi repository does not explicitly give the train set, so we have to remove the test set from qulac.json to get the training data
qulac_train_set = []
qulac_dev_set = []

idk_list = [
    'i dont know',
    'i do not know',
    'im not sure',
    'i am not sure',
    'unsure',
    'possibly',
    'this is not related to my search'
    ]

negation_list = [
    'no',
    'not',
    'none',
    'isnt',
    'isn\'t',
    'dont',
    'don\'t',
    ]


auxiliary_verb_list = [
    'did',
    'do',
    'does',
    'it',
    'are',
    'was',
    'were',
    'have',
    'has',
    'can',
    'could',
    'will',
    'would',
]

def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = text.lower()
    text = re.sub("'(.*)'", r"\1", text)
    if text[-1] == '?':
        text = text[:-1] + '.'
    return text

def type_answer(answer):
    if answer in idk_list:
        return 'idk'
    elif 'yes' in answer.split()[:3]:
        return 'yes'
    elif any([w in answer.split()[:3] for w in negation_list]):
        return 'no'
    else:
        return 'open'

def u2i(text):
    text = re.sub('are you', 'am i', text)
    text = re.sub('you', 'i', text)
    return text

def type_question(question):
    if 'or' in question.split():
        return 'or'
    if spell(question.split()[0]) in auxiliary_verb_list:
        return 'yn'
    else:
        return 'open'


device = T.device("cuda")

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained("output/roberta-qulac/checkpoint-102")
local_dir = "./output/t5-small-qulac-long/"
tokenizer = T5Tokenizer.from_pretrained(local_dir)
model = T5ForConditionalGeneration.from_pretrained(local_dir).cuda()

dataset = '../cosearcher/data/qulac.train.json'
qulac_train = pd.read_json(dataset)
qulac_train.replace(['', "NaN", 'NaT'], np.nan, inplace = True)
qulac_train.dropna(subset=['question', 'facet_desc' , 'answer'], how='any', inplace=True)
qulac_train = qulac_train[['topic','facet_desc','question','answer']].copy(deep=True)

for iter, row in qulac_train.iterrows():
    query = normalize_text(qulac_train.at[iter, 'topic'])
    facet = normalize_text(qulac_train.at[iter, 'facet_desc'])
    question = normalize_text(qulac_train.at[iter, 'question'])
    answer = normalize_text(qulac_train.at[iter, 'answer'])
    qulac_train.at[iter, 't5-question'] = facet + ' . ' + query + ' . ' + question
    qulac_train.at[iter, 'unifiedqa-question'] = u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_train.at[iter, 'unifiedqa-question-1'] = "Answer the following question in " + str(len(answer.split())) +  " words : " + u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_train.at[iter, 'unifiedqa-question-10'] = "Answer the following question in " + str(len(answer.split())) +  " words : " + u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_train.at[iter, 'unifiedqa-question-3'] = "Answer the following question in " + str(len(answer.split())) +  " words : " + u2i(question) + ' ? \\n ' + 'i am looking for ' + facet     
    qulac_train.at[iter, 'flan-question'] = 'I am looking for ' + facet  + " \\n Answer the following question and explain why: " + u2i(question) + ' ? '
    qulac_train.at[iter, 'answer'] = normalize_text(answer)
    qulac_train.at[iter, 'answer-type'] = type_answer(answer)
    qulac_train.at[iter, 'question-type'] = type_question(question)
    qulac_train.at[iter, 'answer-len'] = len(answer.split())

    clf_inputs = roberta_tokenizer(qulac_train.at[iter, 'unifiedqa-question'], return_tensors="pt")
    with T.no_grad():
        logits = roberta_model(**clf_inputs).logits
    predicted_class_id = logits.argmax().item()
    prefix = ''
    if predicted_class_id == 3 : # 3: yes
        prefix = 'yes'
    elif predicted_class_id == 1: #1: no
        prefix = 'no'
    elif predicted_class_id == 0: # 0: idk
        prefix = 'i dont know'
    else:
        prefix = ''
    qulac_train.at[iter, 'decoder-input'] = prefix


qulac_train.to_csv('qulac_train.csv')
qulac_train_yn = qulac_train.loc[qulac_train['question-type'] == 'yn']
qulac_train_yn.to_csv('qulac_train_yn.csv')

qulac_train_short = qulac_train[qulac_train['answer-len'] <= 3 ]
qulac_train_long = qulac_train[qulac_train['answer-len'] > 3 ]
qulac_train_short.to_csv('qulac_train_short.csv')
qulac_train_long.to_csv('qulac_train_long.csv')


dataset = '../cosearcher/data/qulac.valid.json'
qulac_dev = pd.read_json(dataset)
qulac_dev.replace(['', "NaN", 'NaT'], np.nan, inplace = True)
qulac_dev.dropna(subset=['question', 'facet_desc' , 'answer'], how='any', inplace=True)
qulac_dev = qulac_dev[['topic','facet_desc','question','answer']].copy(deep=True)

for iter, row in qulac_dev.iterrows():
    query = normalize_text(qulac_dev.at[iter, 'topic'])
    facet = normalize_text(qulac_dev.at[iter, 'facet_desc'])
    question = normalize_text(qulac_dev.at[iter, 'question'])
    answer = normalize_text(qulac_dev.at[iter, 'answer'])
    qulac_dev.at[iter, 't5-question'] = facet + ' . ' + query + ' . ' + question
    qulac_dev.at[iter, 'unifiedqa-question'] = u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_dev.at[iter, 'unifiedqa-question-1'] = "Answer the following question in " + str(1) +  " words : " + u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_dev.at[iter, 'unifiedqa-question-10'] = "Answer the following question in " + str(10) +  " words : " + u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_dev.at[iter, 'unifiedqa-question-3'] = "Answer the following question in " + str(3) +  " words : " + u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_dev.at[iter, 'flan-question'] = 'I am looking for ' + facet  + " \\n Answer the following question and explain why: " + u2i(question) + ' ? '
    qulac_dev.at[iter, 'answer'] = normalize_text(answer)
    qulac_dev.at[iter, 'answer-type'] = type_answer(answer)
    qulac_dev.at[iter, 'question-type'] = type_question(question)
    qulac_dev.at[iter, 'answer-len'] = len(answer.split())

    clf_inputs = roberta_tokenizer(qulac_dev.at[iter, 'unifiedqa-question'], return_tensors="pt")
    with T.no_grad():
        logits = roberta_model(**clf_inputs).logits
    predicted_class_id = logits.argmax().item()
    prefix = ''
    if predicted_class_id == 3 : # 3: yes
        prefix = 'yes'
    elif predicted_class_id == 1: #1: no
        prefix = 'no'
    elif predicted_class_id == 0: # 0: idk
        prefix = 'i dont know'
    else:
        prefix = ''
    qulac_dev.at[iter, 'decoder-input'] = prefix

qulac_dev.to_csv('qulac_dev.csv')
qulac_dev_yn = qulac_dev.loc[qulac_dev['question-type'] == 'yn']
qulac_dev_yn.to_csv('qulac_dev_yn.csv')

qulac_dev_short = qulac_dev[qulac_dev['answer-len'] <= 3 ]
qulac_dev_long = qulac_dev[qulac_dev['answer-len'] > 3 ]
qulac_dev_short.to_csv('qulac_dev_short.csv')
qulac_dev_long.to_csv('qulac_dev_long.csv')


dataset = '../cosearcher/data/qulac.test.json'
qulac_test = pd.read_json(dataset)
qulac_test.replace(['', "NaN", 'NaT'], np.nan, inplace = True)
qulac_test.dropna(subset=['question', 'facet_desc' , 'answer'], how='any', inplace=True)
qulac_test = qulac_test[['topic','facet_desc','question','answer']].copy(deep=True)

for iter, row in qulac_test.iterrows():
    query = normalize_text(qulac_test.at[iter, 'topic'])
    facet = normalize_text(qulac_test.at[iter, 'facet_desc'])
    question = normalize_text(qulac_test.at[iter, 'question'])
    answer = normalize_text(qulac_test.at[iter, 'answer'])
    qulac_test.at[iter, 't5-question'] = facet + ' . ' + query + ' . ' + question
    qulac_test.at[iter, 'unifiedqa-question'] = u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_test.at[iter, 'unifiedqa-question-1'] = "Answer the following question in " + str(1) +  " words : " + u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_test.at[iter, 'unifiedqa-question-10'] = "Answer the following question in " + str(10) +  " words : " + u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_test.at[iter, 'unifiedqa-question-3'] = "Answer the following question in " + str(3) +  " words : " + u2i(question) + ' ? \\n ' + 'i am looking for ' + facet 
    qulac_test.at[iter, 'flan-question'] = 'I am looking for ' + facet  + " \\n Answer the following question and explain why: " + u2i(question) + ' ? '
    qulac_test.at[iter, 'answer'] = normalize_text(answer)
    qulac_test.at[iter, 'answer-type'] = type_answer(answer)
    qulac_test.at[iter, 'question-type'] = type_question(question)
    qulac_test.at[iter, 'answer-len'] = len(answer.split())

    clf_inputs = roberta_tokenizer(qulac_test.at[iter, 'unifiedqa-question'], return_tensors="pt")
    with T.no_grad():
        logits = roberta_model(**clf_inputs).logits
    predicted_class_id = logits.argmax().item()
    prefix = ''
    if predicted_class_id == 3 : # 3: yes
        prefix = 'yes'
    elif predicted_class_id == 1: #1: no
        prefix = 'no'
    elif predicted_class_id == 0: # 0: idk
        prefix = 'i dont know'
    else:
        prefix = ''
    qulac_test.at[iter, 'decoder-input'] = prefix

qulac_test.to_csv('qulac_test.csv')
qulac_test_yn = qulac_test.loc[qulac_test['question-type'] == 'yn']
qulac_test_yn.to_csv('qulac_test_yn.csv')

qulac_test_short = qulac_test[qulac_test['answer-len'] <= 3 ]
qulac_test_long = qulac_test[qulac_test['answer-len'] > 3 ]
qulac_test_short.to_csv('qulac_test_short.csv')
qulac_test_long.to_csv('qulac_test_long.csv')

print("success!")