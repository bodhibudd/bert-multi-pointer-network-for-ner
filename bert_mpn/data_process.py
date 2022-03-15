import codecs, json
from tqdm import tqdm
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese/")
with codecs.open("data/dev_data.json", 'r', encoding='utf_8_sig') as f:
    gold_num = 0
    p_id = 0
    count = 0
    total_count = 0
    for line in tqdm(f):
        p_id += 1
        data_json = json.loads(line.strip())
        text = data_json['originalText']
        bert_tokens = tokenizer.tokenize(text)
        total_count += 1
        if len(bert_tokens)>512:
            count += 1
    print( count / total_count)
