import os
import sys
import pickle
import json
import copy
import gc
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from util import *
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


topN = 25
DEBUG = False
model_path = sys.argv[1]
model_version = sys.argv[2]
lora_path = sys.argv[3]
batch_size = 32
max_length = 512
output_path = './output/'
data_path = '../input/eedi-mining-misconceptions-in-mathematics/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'model_path = {model_path}')
print(f'lora_path  = {lora_path}')
os.makedirs(output_path, exist_ok=True)

task_description = 'Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception.'


df = pd.read_csv(data_path + 'train.csv')
misconception_mapping = pd.read_csv(data_path + 'misconception_mapping.csv')

misconception_mapping['query_text'] = misconception_mapping['MisconceptionName']
misconception_mapping['order_index'] = misconception_mapping['MisconceptionId']

misconception_mapping['MisconceptionId'] = misconception_mapping['MisconceptionId'].map(lambda x: int(x))
misconception_mapping_dic = misconception_mapping.set_index('MisconceptionId')['MisconceptionName'].to_dict()


tokenizer = AutoTokenizer.from_pretrained(model_path)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
model = AutoModel.from_pretrained(model_path, quantization_config=bnb_config, device_map=device)

if lora_path != 'none':
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
        ],
        bias='none',
        lora_dropout=0.05,  # Conventional
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, config)
    d = torch.load(lora_path, map_location=model.device)
    model.load_state_dict(d, strict=False)
    model = model.merge_and_unload()

model = model.eval()


# 打印模型的设备信息
def print_model_device(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Device: {param.device}")
        break

print_model_device(model)


sentence_embeddings = inference(misconception_mapping, model, tokenizer, device, batch_size, max_length)

index_paper_text_embeddings_index = {index: paper_id for index, paper_id in enumerate(list(sentence_embeddings.keys()))}
sentence_embeddings = np.concatenate([e.reshape(1, -1) for e in list(sentence_embeddings.values())])
sentence_embeddings_tensor = torch.tensor(sentence_embeddings).to(device)


# 构建query
data = []
for _, row in df.iterrows():
    for c in ['A', 'B', 'C', 'D']:
        if str(row[f'Misconception{c}Id']) != 'nan':
            # print(row[f'Misconception{c}Id'])
            real_answer_id = row['CorrectAnswer']
            real_text = row[f'Answer{real_answer_id}Text']
            query_text = f"###question###:{row['SubjectName']}-{row['ConstructName']}-{row['QuestionText']}\n###Correct Answer###:{real_text}\n###Misconcepte Incorrect answer###:{c}.{row[f'Answer{c}Text']}"
            row['query_text'] = get_detailed_instruct(task_description, query_text)
            row['answer_id'] = [int(row[f'Misconception{c}Id'])]
            data.append(copy.deepcopy(row))
data = pd.DataFrame(data)

# 划分CV
train = data[data['QuestionId'] % 5 != 0].reset_index(drop=True)
valid = data[data['QuestionId'] % 5 == 0].reset_index(drop=True)
train['order_index'] = list(range(len(train)))
valid['order_index'] = list(range(len(valid)))
print(f'train.shape = {train.shape} valid.shape = { valid.shape}')


# query embedding
query_embedding_train = inference(train, model, tokenizer, device, batch_size, max_length)
query_embedding_valid = inference(valid, model, tokenizer, device, batch_size, max_length)
# del model
gc.collect()
torch.cuda.empty_cache()

RANK = 100
predict_train = get_predict(train, query_embedding_train, sentence_embeddings_tensor, index_paper_text_embeddings_index, device, RANK=RANK)
train['top_recall_pids'] = predict_train

# 构建负样本，即在召回结果中，去除正确答案
train['new_had_recall_pids'] = list(map(lambda x, y: remove_duplication(x, y), train['top_recall_pids'], train['answer_id']))

# 通过 id 得到文本
train['new_had_recall_ctxs'] = train['new_had_recall_pids'].apply(lambda x: recall_context(x, misconception_mapping_dic))
train['new_positive_ctxs'] = train['answer_id'].apply(lambda x: recall_context(x, misconception_mapping_dic))

# 计算召回率
# train['recalled'] = list(map(lambda x, y: is_recall(x, y, topN), train['top_recall_pids'], train['answer_id']))
train['recalled'] = train.apply(lambda x: is_recall(x['top_recall_pids'], x['answer_id'], topN), axis=1)
# print(train.head())
print(f"train recall rate = {np.mean(train['recalled'])}")


# 计算mapk
actual = train['answer_id'].to_list()
predicted = train['top_recall_pids'].to_list()
train_map_score = mapk(actual, predicted, k=25)
print(f'train mapk score = {train_map_score}')


RANK = 100
predict_valid = get_predict(valid, query_embedding_valid, sentence_embeddings_tensor, index_paper_text_embeddings_index, device, RANK=RANK)
valid['top_recall_pids'] = predict_valid


# 构建负样本，即去重召回中，包含正确答案的数据
valid['new_had_recall_pids'] = list(map(lambda x, y: remove_duplication(x, y), valid['top_recall_pids'], valid['answer_id']))

valid['new_had_recall_ctxs'] = valid['new_had_recall_pids'].apply(lambda x: recall_context(x, misconception_mapping_dic))
valid['new_positive_ctxs'] = valid['answer_id'].apply(lambda x: recall_context(x, misconception_mapping_dic))

# 计算召回率
# valid['recalled'] = list(map(lambda x, y: is_recall(x, y, topN), valid['top_recall_pids'], valid['answer_id']))
valid['recalled'] = valid.apply(lambda x: is_recall(x['top_recall_pids'], x['answer_id'], topN), axis=1)
print(f"valid recall rate = {np.mean(valid['recalled'])}")

actual = valid['answer_id'].to_list()
predicted = valid['top_recall_pids'].to_list()
valid_mapk_score = mapk(actual, predicted, k=topN)
print(f'valid mapk score = {valid_mapk_score}')


# 召回
cnt = 100
bge_recall_train = get_bge_recall(train, cnt=cnt)
print(f'len(bge_recall_train) = {len(bge_recall_train)}')
with open(f'{output_path}{model_version}_recall_top_{cnt}_train.jsonl', 'w') as f:
    f.write('\n'.join([json.dumps(d, ensure_ascii=False) for d in bge_recall_train]))

cnt = 100
bge_recall_valid = get_bge_recall(valid, cnt=cnt)
print(f'len(bge_recall_valid) = {len(bge_recall_valid)}')
with open(f'{output_path}{model_version}_recall_top_{cnt}_valid.jsonl', 'w') as f:
    f.write('\n'.join([json.dumps(d, ensure_ascii=False) for d in bge_recall_valid]))


# 排序
cnt = 50
bge_rank_train = get_bge_recall(train, cnt=cnt)
print(f'len(bge_rank_train) = {len(bge_rank_train)}')
with open(f'{output_path}{model_version}_rank_top_{cnt}_train.jsonl', 'w') as f:
    f.write('\n'.join([json.dumps(d, ensure_ascii=False) for d in bge_rank_train]))

cnt = 50
bge_rank_valid = get_bge_recall(valid, cnt=cnt)
print(f'len(bge_rank_valid) = {len(bge_rank_valid)}')
with open(f'{output_path}{model_version}_rank_top_{cnt}_valid.jsonl', 'w') as f:
    f.write('\n'.join([json.dumps(d, ensure_ascii=False) for d in bge_rank_valid]))
