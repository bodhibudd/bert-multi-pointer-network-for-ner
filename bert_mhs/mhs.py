from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
import numpy as np
from config import MHS_ENTITY
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
class NERNet(BertPreTrainedModel):
    def __init__(self, config, class_num = None):
        super(NERNet, self).__init__(config, class_num)
        self.class_num = class_num
        self.config = config
        self.bert = BertModel(config)
        self.biaffine_linar = torch.nn.Parameter(torch.randn(128+1, class_num, 128+1))
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=768, \
                                  num_layers=1, batch_first=True, \
                                  dropout=0.5, bidirectional=True)
        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*768, out_features=128),
                                               torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*768, out_features=128),
                                             torch.nn.ReLU())
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def forward(self, q_ids=None, input_ids=None, segment_ids = None, entity_labels=None, eval_file=None, is_eval=False, add_bias=True):

        attention_mask = input_ids != 0
        batch_mask = []
        for batch in range(attention_mask.shape[0]):
            mask = [
                attention_mask[batch].cpu().numpy() if i == 1 else torch.zeros(attention_mask.shape[1]).cpu().numpy()
                for i in attention_mask[batch]]
            batch_mask.append(mask)
        batch_mask = torch.tensor(batch_mask).to(device)
        bert_encoder = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,
                                 output_hidden_states=True).hidden_states
        lstm_encoder, _ = self.lstm(bert_encoder[-1])
        ffnn_start = self.start_layer(lstm_encoder)
        ffnn_end = self.end_layer(lstm_encoder)
        ffnn_start = torch.cat(
            [ffnn_start, torch.ones((bert_encoder[-1].shape[0], input_ids.shape[1], 1), device=device)], dim=2)
        ffnn_end = torch.cat(
            [ffnn_end, torch.ones((bert_encoder[-1].shape[0], input_ids.shape[1], 1), device=device)], dim=2)
        biaffine = torch.einsum('bih,hry,bjy->bijr', ffnn_start, self.biaffine_linar, ffnn_end)
        if not is_eval:

            #shape [b,l,l,c]
            biaffine = biaffine.contiguous()
            biaffine = biaffine.view(size=(-1, self.class_num))
            entity_labels = entity_labels.view(size=(-1,))
            batch_mask = batch_mask.view(size=(-1,))

            loss = self.loss_fct(biaffine, entity_labels)
            loss *= batch_mask
            loss = torch.sum(loss) / batch_mask.size()[0]
            return loss
        ent_preds = nn.Softmax(dim=-1)(biaffine)
        ent_labels = []
        seq_labels = []
        qid_set = set(q_ids.cpu().numpy())
        count = 0
        for qid in qid_set:
            hz_examples = [example for example in eval_file if example.p_id == qid + 1]
            hz_ent_pred = ent_preds.cpu().numpy()[count:count + len(hz_examples)]
            count = len(hz_examples)
            for example, ent_pred in zip(hz_examples, hz_ent_pred):
                # l*c
                context = example.bert_tokens
                ent_label = []
                for i, s in enumerate(ent_pred):
                    preds = np.max(s, axis=-1)
                    pred_types = np.argmax(s, axis=-1)
                    for j, (pred, pred_type) in enumerate(zip(preds, pred_types)):
                        if pred_type != 0 and i <= j:
                            ent_label.append((i, j, pred_type, pred))

                # 排序ent_label，以防解码非最优
                ent_label = sorted(ent_label, key=lambda x: x[3], reverse=True)
                ent_labels.append(ent_label)
                seq_label = []
                for e_label in ent_label:
                    if e_label[0] > len(context) - 2 or e_label[0] == 0:
                        continue
                    if e_label[1] > len(context) - 2:
                        continue
                    for s_label in seq_label:
                        if e_label[0] <= s_label[0] <= e_label[1] <= s_label[1] or s_label[0] <= e_label[0] <= s_label[
                            1] <= e_label[1]:
                            break
                    else:
                        seq_label.append((e_label[0], e_label[1], e_label[2], example.token_ids[e_label[0]:e_label[1]+1]))
                seq_label = sorted(seq_label, key=lambda x: x[0], reverse=False)

                seq_labels.append((qid.item(), seq_label))

        return seq_labels
        # for q_id, ent_pred in zip(q_ids.cpu().numpy(), ent_preds.cpu().numpy()):
        #     ent_label = []
        #     #l*c
        #     context = eval_file[q_id.item()].bert_tokens
        #     for i, s in enumerate(ent_pred):
        #         preds = np.max(s, axis=-1)
        #         pred_types = np.argmax(s, axis=-1)
        #         for j, (pred, pred_type) in enumerate(zip(preds, pred_types)):
        #             if pred_type != 0 and i <=j:
        #                 ent_label.append((i, j, pred_type, pred))
        #
        #     #排序ent_label，以防解码非最优
        #     ent_label = sorted(ent_label, key = lambda x: x[3], reverse=True)
        #     ent_labels.append(ent_label)
        #     seq_label = []
        #     for e_label in ent_label:
        #         if e_label[0] > len(context)-2 or e_label[0] == 0:
        #             continue
        #         if e_label[1] > len(context)-2:
        #             continue
        #         for s_label in seq_label:
        #             if e_label[0] <= s_label[0]<= e_label[1]<=s_label[1] or s_label[0]<=e_label[0]<=s_label[1]<= e_label[1]:
        #                 break
        #         else:
        #             seq_label.append((e_label[0], e_label[1], e_label[2]))
        #     seq_label = sorted(seq_label, key=lambda x: x[0], reverse=False)
        #
        #     seq_labels.append((q_id.item(), seq_label))

