import codecs
import json
import logging
import random
from collections import Counter
from functools import partial
from utils.data_util import Tokenizer
from transformers import BertTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from config import MPN_ENTITY

class Example(object):
    def __init__(self,
                 p_id=None,
                 context=None,
                 bert_tokens=None,
                 entity_list=None,
                 token_ids=None):
        self.p_id = p_id
        self.context = context
        self.bert_tokens = bert_tokens
        self.entity_list = entity_list
        self.tokens_ids = token_ids

class Reader(object):
    def __init__(self,do_lowercase=False):
        self.do_lowercase = do_lowercase

    def read_examples(self, filename,data_type):
        examples = []
        ent2id = {ent:id for id, ent in enumerate(MPN_ENTITY)}
        with codecs.open(filename,'r',encoding='utf_8_sig') as f:
            gold_num = 0
            p_id = 0
            for line in tqdm(f):
                p_id += 1
                data_json = json.loads(line.strip())
                text = data_json['originalText']
                ent_list = list()

                for entity in data_json['entities']:
                    entity_type = entity["label_type"]
                    start_pos = entity['start_pos']
                    end_pos = entity['end_pos']
                    entity_name = text[entity['start_pos']:entity['end_pos']]
                    ent_list.append((entity_name, ent2id[entity_type], start_pos, end_pos-1))
                # text = data_json['originalText'].strip().lower()
                examples.append(
                    Example(p_id=p_id,
                            context=text,
                            entity_list=ent_list)
                )
                gold_num += len(ent_list)
        print("total gold num is {}".format(gold_num))

        logging.info("{} total size is {}".format(data_type, len(examples)))

        return examples

class Feature(object):
    def __init__(self, args):
        self.max_len = args.max_len
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    def __call__(self, examples, data_type):
        return self.convert_examples_to_bert_features(examples, data_type)

    def convert_examples_to_bert_features(self, examples, data_type):
        examples2features = list()
        for index, example in enumerate(examples):
            examples2features.append((index, example))
        return SPODataset(examples2features, data_type, self.tokenizer)
class SPODataset(Dataset):
    def __init__(self, data, data_type, tokenizer=None, max_len=512):
        super(SPODataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.q_ids = [f[0] for f in data]
        self.features = [f[1] for f in data]
        self.train = True if data_type == "train" else False

    def __len__(self):
        return len(self.q_ids)
    def __getitem__(self, index):
        return self.q_ids[index], self.features[index]

    def search(self, text_ids, target_ids, start=None, end=None):
        n = len(target_ids)
        if len(target_ids) == 0:
            return -1
        if start != None:
            if start > len(text_ids):
                return -1
            for i in range(start, len(text_ids)):
                if (text_ids[i:i + n]) == target_ids:
                    return i
        else:
            for i in range(len(text_ids)):
                if(text_ids[i:i+n]) == target_ids:
                    return i
        return -1
    def _create_collate_fn(self):
        def collate(examples):
            p_ids, examples = zip(*examples)
            #input_ids,segment_id,attention_mask
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            batch_token_ids, batch_segment_ids, batch_entity_labels = [], [], []
            max_len = max([len(self.tokenizer.tokenize(example.context)) for example in examples])
            if max_len > self.max_len:
                max_len = self.max_len
            for example in examples:
                encoder = self.tokenizer(example.context, max_length=max_len)
                token_ids, segment_ids = encoder["input_ids"], encoder["token_type_ids"]
                #通过排序的手段解决对齐问题
                example.bert_tokens = self.tokenizer.tokenize(example.context)
                example.token_ids = token_ids
                #第一次匹配的pos位置
                aim_pos = 0
                ent_label = np.zeros((len(token_ids), len(MPN_ENTITY), 2))
                for entity in example.entity_list:
                    entity_ids = self.tokenizer.encode(entity[0])[1:-1]#去掉CLS和SEP
                    start_pos = self.search(token_ids, entity_ids, aim_pos)
                    if start_pos < 0:
                        continue
                    aim_pos = start_pos+len(entity_ids)
                    ent_label[start_pos, entity[1], 0] = 1 #加1是由于[CLS]
                    ent_label[start_pos+len(entity_ids)-1, entity[1], 1] = 1

                batch_entity_labels.append(ent_label)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)

            #seqpadding
            batch_token_ids = seqpadding(batch_token_ids, 0)
            batch_segment_ids = seqpadding(batch_segment_ids, 0)
            if not self.train:
                return p_ids, batch_token_ids, batch_segment_ids

            batch_entity_labels = seqpadding(batch_entity_labels, np.zeros((len(MPN_ENTITY), 2)), is_float= True)

            return batch_token_ids, batch_segment_ids, batch_entity_labels
        return partial(collate)
    def get_dataloader(self, batch_size, shuffle=False, pin_memory=False, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(), pin_memory=pin_memory, num_workers=num_workers)

def seqpadding(datas, padding, max_len=None, is_float = False):
    if max_len is None:
        max_len = max([len(data) for data in datas])
    outputs = [
        np.concatenate([x, [padding] * (max_len - len(x))])
        if len(x) < max_len else x[:max_len] for x in datas
    ]
    output_tensors = torch.FloatTensor(outputs) if is_float else torch.tensor(outputs)
    return output_tensors

def search(sids, token_ids):
    if(len(token_ids) < len(sids)):
        return -1
    index = -1
    for i, id in enumerate(token_ids):
        if(token_ids[i:i+len(sids)] == sids):
            return i
    return index



