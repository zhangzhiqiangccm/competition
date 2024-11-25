import json
import numpy as np
from tqdm import tqdm, trange
import torch
from torch import Tensor
import torch.nn.functional as F


def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.
    
    This function computes the average prescision at k between two lists of
    items.
    
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]
        
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.
    
    This function computes the mean average prescision at k between two lists
    of lists of items.
    
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
        
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """    
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def is_recall(x, y, topN=25):
    if y[0] in x[:topN]:
        return 1
    else:
        return 0


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def inference(df, model, tokenizer, device, batch_size=16, max_length=512):
    pids = list(df['order_index'].values)
    sentences = list(df['query_text'].values)
    
    # 根据文本长度逆序：长的在前，短的在后
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    # print(length_sorted_idx[:5])
    # print(sentences_sorted[:5])
    
    all_embeddings = []
    for start_index in trange(0, len(sentences), batch_size, desc='Batches', disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        # if start_index == 0:
        #     print(f'features = {features.keys()}')
        # features input_id, attention_mask
        features = batch_to_device(features, device)
        with torch.no_grad():
            outputs = model(**features)
            embeddings = last_token_pool(outputs.last_hidden_state, features['attention_mask'])
            # if start_index == 0:
            #     print(f'embeddings = {embeddings.detach().cpu().numpy().shape}')
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.detach().cpu().numpy().tolist()
        all_embeddings.extend(embeddings)
    
    all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]
    
    sentence_embeddings = np.concatenate(all_embeddings, axis=0)
    result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
    return result


def remove_duplication(x, y):
    res = []
    for xi in x:
        if xi not in y:
            res.append(xi)
    return res


def recall_context(x, misconception_mapping_dic):
    res = []
    for xi in x:
        MisconceptionName = misconception_mapping_dic[xi]
        res.append({'MisconceptionName': MisconceptionName})
    return res


def get_predict(df, query_embedding, sentence_embedding, index_paper_text_embeddings_index, device, RANK=100):
    predict_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        query_id = row['order_index']
        query_em = query_embedding[query_id].reshape(1, -1)
        query_em = torch.tensor(query_em).to(device).view(1, -1)
        
        score = (query_em @ sentence_embedding.T).squeeze()
        
        sort_index = torch.sort(-score).indices.detach().cpu().numpy().tolist()[:RANK]
        pids = [index_paper_text_embeddings_index[index] for index in sort_index]
        predict_list.append(pids)
    return predict_list


def get_bge_recall(df, cnt=100):
    bge_recall = []
    for _, row in df.iterrows():
        query = row['query_text']
            
        # 正样本
        pos = []
        for data in row['new_positive_ctxs']:
            pos.append(data['MisconceptionName'])
        
        # 负样本
        neg = []
        hard_negative_ctxs = row['new_had_recall_ctxs'][:cnt]
        for data in hard_negative_ctxs:
            neg.append(data['MisconceptionName'])
        
        bge_recall.append({'query': query, 'pos': pos, 'neg': neg})
    return bge_recall
    

def get_bge_rank(df, cnt=50):
    bge_rank = []
    for _, row in df.iterrows():
        query = row['query_text']
        
        # 正样本
        pos = []
        for data in row['new_positive_ctxs']:
            pos.append(data['MisconceptionName'])
        
        # 负样本
        neg = []
        hard_negative_ctxs = row['new_had_recall_ctxs'][:cnt]
        for data in hard_negative_ctxs:
            neg.append(data['MisconceptionName'])
        
        bge_rank.append({
            'query': query,
            'pos': pos,
            'neg': neg,
            'prompt': "Given a query with a SubjectName, along with a ConstructName, QuestionText, CorrectAnswer, and Misconcepte Incorrect Answer, determine whether the Misconcepte Incorrect Answer is pertinent to the query by providing a prediction of either 'Yes' or 'No'."
        })
    return bge_rank
