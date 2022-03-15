# from utils.data_util import Tokenizer
from transformers import BertTokenizer
import torch
from bert_mpn import mpn
from utils.optimize_util import set_optimize
from tqdm import tqdm
import sys
import numpy as np
class Trainer(object):

    def __init__(self, args, data_loaders, examples, ent_conf):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        if self.n_gpu > 0:
            torch.cuda.manual_seed(args.seed)
        self.id2ent = ent_conf
        self.ent2id = {ent:id for id, ent in enumerate(ent_conf)}
        self.model = mpn.NERNet.from_pretrained(args.bert_model, class_num = len(ent_conf))
        self.model.to(self.device)
        self.train_loader, self.dev_loader = data_loaders
        self.train_examples, self.dev_examples = examples
        self.optimezer = set_optimize(args, model=self.model, train_step=(int(len(self.train_examples) / args.train_batch_size) + 1) * args.epoch_num)

    def train(self, args):
        best_f1 = 0.0
        patience_stop = 0
        self.model.train()
        step_gap = 20

        for epoch in range(int(args.epoch_num)):
            global_loss = 0.0
            for step, batch in tqdm(enumerate(self.train_loader), mininterval=5,
                                    desc=u'training at epoch : %d ' % epoch, leave=False, file=sys.stdout):
                batch = tuple(t.to(self.device) for t in batch)
                loss = self.forward(batch)
                global_loss += loss
                if step % step_gap == 0:
                    current_loss = global_loss / step_gap
                    print(
                        u"step {} / {} of epoch {}, train/loss: {}".format(step, len(self.train_loader),
                                                                           epoch, current_loss))
                    global_loss = 0.0

                if step % 500 == 0 and epoch >= 6:
                    res_dev = self.eval_data_set("dev")
                    if res_dev["f1"] > best_f1:
                        best_f1 = res_dev["f1"]
                        save_model = self.model.module if hasattr(self.model, "module") else self.model
                        output_path = args.output + "pytorch_model.bin"
                        torch.save(save_model.state_dict(), output_path)
                        patience_stop = 0
                else:
                    patience_stop += 1
                if patience_stop > args.patience_stop:
                    return
            res_dev = self.eval_data_set("dev")
            if res_dev["f1"] > best_f1:
                best_f1 = res_dev["f1"]

                save_model = self.model.module if hasattr(self.model, "module") else self.model
                output_path = args.output + "pytorch_model.bin"
                torch.save(save_model.state_dict(), output_path)
                patience_stop = 0
            else:
                patience_stop += 1
            if patience_stop >= args.patience_stop:
                return

    def forward(self, batch, eval=False, answer_dict=None):
        input_ids, segment_ids, ent_labels = batch
        if not eval:
            loss = self.model(passage_ids=input_ids, segment_ids=segment_ids,entity_labels=ent_labels)
            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            loss.backward()
            loss = loss.item()
            self.optimezer.step()
            self.optimezer.zero_grad()
            return loss

        p_ids, input_ids, segment_ids = batch
        eval_file = self.dev_examples
        ent_list = self.model(q_ids=p_ids, passage_ids=input_ids, segment_ids=segment_ids, eval_file = eval_file, is_eval=True)

        ans_dict = self.convert_entities(eval_file, ent_list)

        answer_dict.update(ans_dict)


    def eval_data_set(self, chosen="dev"):
        self.model.eval()
        eval_file = self.dev_examples
        answer_dict = {i :{} for i in range(len(eval_file))}
        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.dev_loader), mininterval=5, leave=False, file=sys.stdout):
                batch = tuple(t.to(self.device) for t in batch)
                self.forward(batch, eval=True, answer_dict=answer_dict)

            self.model.train()
            res = self.evaluate(eval_file, answer_dict, chosen)
            return res

    def resume(self, args):
        resume_model_file = args.output + "/pytorch_model.bin"
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def show(self, chosen='dev'):
        self.model.eval()
        data_loader = self.dev_loader
        eval_file = self.dev_examples
        answer_dict = {i: {} for i in range(len(eval_file))}
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                batch = tuple(t.to(self.device) for t in batch)
                self.forward(batch, eval=True, answer_dict=answer_dict)
            print(answer_dict)

    def evaluate(self, eval_file, answer_dict, chosen):
        entity_em = 0
        entity_pred_em = 0
        entity_gold_em = 0
        for i, answer in answer_dict.items():
            entity_list = eval_file[i].entity_list
            entity_gold = [(ent[0], ent[1]) for ent in entity_list]
            entity_pred = [(ent[0], self.ent2id[ent[1]]) for ent in answer.get("entities",[])]
            entity_em += len(set(entity_pred)&set(entity_gold))
            entity_pred_em += len(set(entity_pred))
            entity_gold_em += len(set(entity_gold))
        entity_precision = 100.0 * entity_em/(entity_pred_em+1e-10)
        entity_recall = 100.0 * entity_em/(entity_gold_em+1e-10)
        entity_f1 = 2*entity_precision*entity_recall/(entity_precision+entity_recall+1e-10)

        print("{}/entity_em: {},\tentity_pred_num&entity_gold_num: {}\t{} ".format(chosen, entity_em, entity_pred_em,
                                                                                   entity_gold_em))
        print(
            "{}/entity_f1: {}, \tentity_precision: {},\tentity_recall: {} ".format(chosen, entity_f1, entity_precision,
                                                                                   entity_recall))
        return {'f1': entity_f1, "recall": entity_recall, "precision": entity_precision}

    def convert_entities(self, eval_file, entities_list):
        answer_dict = {}
        for qid, ent_list in entities_list:
            # tokens = eval_file[qid].bert_tokens
            token_ids = eval_file[qid].token_ids
            text = eval_file[qid].context
            answer_dict[qid]={}
            answer_dict[qid]["text"] = eval_file[qid].context
            answer_dict[qid]["entities"] = []
            aim_pos = 0
            for ent in ent_list:
                entity_name = self.tokenizer.decode(token_ids=token_ids[ent[0]:ent[1]+1])
                entity_name = entity_name.replace(" ","")
                entity_type = self.id2ent[ent[2]]
                start_pos = self.search(text, entity_name, start=aim_pos)
                if start_pos < 0:
                    continue
                aim_pos = start_pos+ len(entity_name)
                answer_dict[qid]["entities"].append((entity_name, entity_type, start_pos, start_pos+len(entity_name)-1))
        # print("-----------" + str(answer_dict) + "------------")
        return answer_dict

    def search(self, text_ids, target_ids, start=None, end=None):
        n = len(target_ids)
        if len(target_ids) == 0:
            return -1
        if start != None:
            for i in range(start, len(text_ids)):
                if (text_ids[i:i + n]) == target_ids:
                    return i
        else:
            for i in range(len(text_ids)):
                if(text_ids[i:i+n]) == target_ids:
                    return i
        return -1


