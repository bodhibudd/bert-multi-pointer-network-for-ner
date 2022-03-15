import argparse
from utils.file_util import save
from config import MHS_ENTITY
from bert_mhs.data_loader import Feature, Reader
from bert_mhs.train import Trainer
import os,json
import pickle
import random, numpy as np
import torch
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/",type=str,required=False)
    parser.add_argument("--output",default="output/",type=str, required=False)
    parser.add_argument("--train_mode",type=str, default="train")
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--dev_batch_size", default=1, type=int)
    parser.add_argument("--learning_rate",default=5e-5,type=float)
    parser.add_argument("--epoch_num", default=30,type=int)
    parser.add_argument("--patient_stop",type=int,default=10000)
    parser.add_argument("--device_id",type=int, default=0)
    parser.add_argument("--seed",type=int, default=42)
    parser.add_argument("--do_lower_case",action='store_true')
    parser.add_argument("--warmup_proportion",default=0.1,type=float)
    parser.add_argument("--warmup", default=True, type=bool)
    parser.add_argument("--bert_model",default="bert-base-chinese/",type=str)
    parser.add_argument("--max_len",default=100,type=int)
    parser.add_argument("--hidden_size",type=int,default=150)
    parser.add_argument("--bert_hidden_size",type=int,default=768)
    parser.add_argument("--dropout",type=float,default=0.5)
    parser.add_argument("--dim_feedforward",type=int,default=2048)
    parser.add_argument("--patience_stop", default=1000, type=int)

    args = parser.parse_args()
    args.cache_data = args.input
    return args


def build_dataset(args, reader, debug=False):

    train_src = args.input+"/dev_data.json"
    dev_src = args.input +"/dev_data.json"

    train_example_file = args.cache_data + "/train-examples.pkl"
    dev_examples_file = args.cache_data + "/dev-examples.pkl"


    if not os.path.exists(train_example_file):
        train_examples = reader.read_examples(train_src,data_type='train')
        dev_examples = reader.read_examples(dev_src,data_type='dev')

        save(train_example_file, train_examples)
        save(dev_examples_file, dev_examples)
    else:
        with open(train_example_file, 'rb') as f1, open(dev_examples_file, 'rb') as f2:
            train_examples = pickle.load(f1)
            dev_examples = pickle.load(f2)
    convert_examples_features = Feature(args)
    train_dataset = convert_examples_features(train_examples, 'train')
    dev_dataset = convert_examples_features(dev_examples, 'dev')

    train_dataloader = train_dataset.get_dataloader(args.train_batch_size, shuffle=False, pin_memory=False)
    dev_dataloader = dev_dataset.get_dataloader(args.dev_batch_size, shuffle=False)
    data_loader = train_dataloader, dev_dataloader
    eval_examples = train_examples, dev_examples
    # with open("data/shujuji.json",'w',encoding="utf8") as f:
    #     for i, example in enumerate(dev_examples):
    #         print("p_id:"+str(example.p_id)+","+"entity_list:"+str(example.entity_list))
    return eval_examples, data_loader

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    reader = Reader(do_lowercase=True)
    eval_examples, data_loader = build_dataset(args, reader)
    trainer = Trainer(args, data_loader, eval_examples, MHS_ENTITY)
    if args.train_mode == "train":
        trainer.train(args)
    elif args.train_mode == 'dev':
        trainer.resume(args)
        trainer.eval_data_set(chosen='dev')
    else:
        trainer.resume(args)
        trainer.show(chosen='dev')

if __name__ == "__main__":
    main()
