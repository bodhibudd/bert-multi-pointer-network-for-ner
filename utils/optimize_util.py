import torch
from torch import optim
from run.entity_relation_jointed_extraction.bert.bert_optimization import BertAdam
def set_optimize(args, model, train_step):
    if args.warmup:
        parameters = list(model.named_parameters())
        param_optimizer = [ parameter for parameter in parameters if "pooler" not in parameter[0]]
        no_decay = ["bias","LayerNorm.bias", "LayerNorm.wight"]
        optimizer_param_groups = [
            {"params": [p for n, p in param_optimizer if not any( nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params":[p for n, p in param_optimizer if any( nd in n for nd in no_decay)], "weight_decay":0.0}
        ]

        optimizer = BertAdam(params=optimizer_param_groups, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=train_step)
    else:
        param_optimizer = list(filter(lambda p:p.requires_grad, model.parameters()))
        optimizer = optim.Adam(param_optimizer, lr= args.learning_rate)
    return optimizer
