from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
import numpy as np
class NERNet(BertPreTrainedModel):
    def __init__(self, config, class_num):
        super(NERNet, self).__init__(config)
        self.bert = BertModel(config)
        self.class_num = class_num
        #subject
        self.entity_dense = nn.Linear(config.hidden_size, self.class_num*2)

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, q_ids=None, passage_ids=None, segment_ids=None,entity_labels=None, eval_file=None,
                is_eval=False):
        mask = passage_ids != 0
        bert_encoder = self.bert(input_ids=passage_ids, token_type_ids=segment_ids, attention_mask = mask, output_hidden_states = False).last_hidden_state

        if not is_eval:
            ent_preds = self.entity_dense(bert_encoder).reshape(-1, passage_ids.size()[1], self.class_num, 2)
            loss = self.loss_fct(ent_preds, entity_labels)
            loss = torch.sum(loss.mean(3),2)
            loss = torch.sum(loss*mask.float()) / torch.sum(mask.float())
            return loss
        else:
            ent_preds = nn.Sigmoid()(self.entity_dense(bert_encoder)).reshape(-1, passage_ids.size()[1], self.class_num, 2)
            entities_dict = []
            for qid, ent_pred in zip(q_ids.cpu().numpy(),
                                     ent_preds.cpu().numpy()):
                context = eval_file[qid.item()].bert_tokens
                ent_list = list()
                #np.where返回的是一个元组，元组的每个元素均为某个维度上对应的索引数组
                starts = np.where(ent_pred[:,:,0] > 0.6)
                ends = np.where(ent_pred[:,:,1] > 0.7)

                for start, start_entity_id in zip(*starts):#解压的意思是将外层剥去
                    if start > len(context) - 2 or start == 0:
                        continue
                    for end, end_entity_id in zip(*ends):
                        if end > len(context) - 2:
                            continue
                        if end >= start and start_entity_id == end_entity_id:
                            ent_list.append((start, end, start_entity_id))
                            break
                entities_dict.append((qid.item(), ent_list))

            return entities_dict
