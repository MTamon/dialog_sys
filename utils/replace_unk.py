import torch
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import numpy as np

import os

config = BertConfig.from_json_file('resource/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/bert_config.json')
model = BertForMaskedLM.from_pretrained('resource/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/pytorch_model.bin', config=config)
bert_tokenizer = BertTokenizer('resource/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM/vocab.txt', do_lower_case=False, do_basic_tokenize=False)
model.eval()

# GPUのセット
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 三項演算子
print("使用デバイス：", device)
model.to(device)

train_path = "data/train_cleaned"
eval_path = "data/eval_cleaned"
output_train_path = "data/train"
output_eval_path = "data/eval"

unk_id = bert_tokenizer.convert_tokens_to_ids('[UNK]')
msk_id = bert_tokenizer.convert_tokens_to_ids('[MASK]')
pad_id = bert_tokenizer.convert_tokens_to_ids('[PAD]')
sep_id = bert_tokenizer.convert_tokens_to_ids('[SEP]')
cls_id = bert_tokenizer.convert_tokens_to_ids('[CLS]')

def data_recreate(in_path, out_path, head_idx=0, batch_size=8, word_replace=False):
    """[UNK]を[MASK]に置換し, BERTのMaskedLanguageModelにて穴埋めする.

    Args:
        in_path (_type_): 入力データのディレクトリ.
        out_path (_type_): 出力データのディレクトリ.
        head_idx (int, optional): 入力データの読み込み開始位置. Defaults to 0.
        batch_size (int, optional): BERTに入力する際のバッチサイズ. Defaults to 8.
        word_replace (bool, opyional): [MASK]以外の単語の置き換えを行うか指定する. Defaults to False.
    """
    files = os.listdir(in_path)
    file_num = len(files)

    file_exists = os.path.exists(out_path)
    if file_exists == False:
        #保存先のディレクトリの作成
        os.mkdir(out_path)

    for i, file_name in enumerate(files):
        path = os.path.join(in_path, file_name)
        print('processing for {} ... '.format(path), end='')
        
        with open(path, mode='r') as f:
            lines = f.read().splitlines()
            
        headders = lines[:head_idx]
        
        querys = []
        answers = []
        for line in lines[head_idx:]:
            # convert str -> int
            query_str, answer_str = line.split(',')
            
            query_str = query_str.split()
            answer_str = answer_str.split()
            
            query = [int(q) for q in query_str]
            answer = [int(a) for a in answer_str]
            
            querys.append(query)
            answers.append(answer)
            
        # shape data
        packs = pack_sequence([torch.tensor(t, device=device).detach() for t in answers], enforce_sorted=False)
        (answers, answers_len) = pad_packed_sequence(
            packs, 
            batch_first=True, 
            padding_value=0.0
        )
        packs = pack_sequence([torch.tensor(t, device=device).detach() for t in querys], enforce_sorted=False)
        (querys, querys_len) = pad_packed_sequence(
            packs, 
            batch_first=True, 
            padding_value=0.0
        )
        
        # setting device
        querys.to(device)
        answers.to(device)
        
        if word_replace:
            # record [PAD] position
            querys_pad = (querys == pad_id)
            answers_pad = (answers == pad_id)
            # record [SEP] position
            querys_sep = (querys == sep_id)
            answers_sep = (answers == sep_id)
            # record [CLS] position
            querys_cls = (querys == cls_id)
            answers_cls = (answers == cls_id)
        
        # record [UNK] position
        querys_unk = (querys == unk_id)
        answers_unk = (answers == unk_id)
        
        # unk -> mask
        querys[querys_unk] = msk_id
        answers[answers_unk] = msk_id
        
        # divide batch (to avoid out of memory)
        querys_div = torch.split(querys, batch_size)
        answers_div = torch.split(answers, batch_size)
        
        # bert masked lm
        new_querys = torch.zeros((1, querys.shape[1]), device=device)
        new_answers = torch.zeros((1, answers.shape[1]), device=device)
        for query_batch, answer_batch in zip(querys_div, answers_div):
            # source mask
            querys_mask = torch.ones(query_batch.shape ,dtype=torch.int32, device=device) * (query_batch != 0)
            answers_mask = torch.ones(answer_batch.shape ,dtype=torch.int32, device=device) * (answer_batch != 0)
            
            # inference
            pre_qry_batch = model(query_batch, attention_mask=querys_mask)
            pre_ans_batch = model(answer_batch, attention_mask=answers_mask)
            
            # collect inference result
            _, qry_batch = torch.topk(pre_qry_batch[0], k=1, dim=2)
            qry_batch = torch.squeeze(qry_batch, dim=2)
            _, ans_batch = torch.topk(pre_ans_batch[0], k=1, dim=2)
            ans_batch = torch.squeeze(ans_batch, dim=2)
            
            # concatnate
            new_querys = torch.cat((new_querys, qry_batch), dim=0)
            new_answers = torch.cat((new_answers, ans_batch), dim=0)
        
        # shape and cast
        new_querys = new_querys[1:].to(torch.int64) # remove zeor tensor
        new_answers = new_answers[1:].to(torch.int64)
        
        if not word_replace:
            querys[querys_unk] = new_querys[querys_unk] # replace only [UNK] position
            answers[answers_unk] = new_answers[answers_unk]
            querys = querys.to(torch.int)
            answers = answers.to(torch.int)
        else:
            querys = new_querys # replace all word
            answers = new_answers
            # recover [CLS] and [SEP] and [PAD]
            querys[querys_cls] = cls_id
            answers[answers_cls] = cls_id
            querys[querys_sep] = sep_id
            answers[answers_sep] = sep_id
            querys[querys_pad] = pad_id
            answers[answers_pad] = pad_id
        
        # write for new file
        path = os.path.join(out_path, file_name)
        with open(path, mode='w') as f:
            # write headder
            if type(headders) == list:
                for headder in headders:
                    f.write(headder + '\n')
            elif type(headders) == str:
                f.write(headders + '\n')
            
            # write data
            for query, answer in zip(querys, answers):
                # remove [PAD]
                qry_pad_mask = (query != pad_id)
                ans_pad_mask = (answer != pad_id)
                query = query[qry_pad_mask]
                answer = answer[ans_pad_mask]
                
                # convert tensor -> str
                query = ' '.join([str(q) for q in query.tolist()])
                answer = ' '.join([str(a) for a in answer.tolist()])
                
                # write record
                f.write('{},{}\n'.format(query, answer))
                
        print('Done.')
if __name__ == '__main__':
    data_recreate(train_path, output_train_path, 1)
    data_recreate(eval_path, output_eval_path, 1)