import re
import unicodedata
import numpy as np
import torch
from transformers import BertTokenizer
def is_string(s):
    return isinstance(s, str)

def padding(seqs, is_float=False, batch_first=False):
    lengths = [len(s) for s in seqs]
    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.FloatTensor(batch_length, len(seqs)).fill_(0) if is_float \
        else torch.LongTensor(batch_length, len(seqs)).fill_(0)
    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq,i].copy_(s)
    if batch_first:
        seq_tensor.t_()

    return seq_tensor, lengths


class BasicTokenizer(object):
    def __init__(self, do_lower_case=False):
        '''
        初始化
        :param do_lower_case:
        '''
        self._token_pad = '[PAD]'
        self._token_cls = '[CLS]'
        self._token_sep = '[SEP]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._do_lower_case = do_lower_case

    def tokenize(self, text, add_cls=True, add_sep=True, max_length=None):
        '''
        分词函数
        :param text:
        :param add_cls:
        :param add_sep:
        :param max_length:
        :return:
        '''
        if self._do_lower_case:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        tokens = self._tokenize(text)
        if add_cls:
            tokens.insert(0,self._token_cls)
        if add_sep:
            tokens.append(self._token_sep)
        if max_length != None:
            self.truncate_sequence(max_length, tokens, None, -2)
        return tokens

    def token_to_id(self, token):
        '''
        token转换为对应的id
        :param token:
        :return:
        '''
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        '''
        token序列转换为对应的id序列
        :param tokens:
        :return:
        '''
        return [self.token_to_id(token) for token in tokens]

    def encode(self,
               first_text,
               second_text=None,
               max_length=None,
               first_length=None,
               second_length=None):
        '''
        输出文本对应token_id和segment_id
        如果传入first_length，则强行padding第一个句子到指定长度；
        如果传入second_length，则强行padding第二个句子到指定长度.
        :param first_text:
        :param second_text:
        :param max_length:
        :param first_length:
        :param second_length:
        :return:
        '''
        if is_string(first_text):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            second_tokens = self.tokenize(second_text, add_cls=False)
        else:
            second_tokens = second_text
        if max_length is not None:
            self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        if first_length is not None:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self._token_pad_id]*(first_length-len(first_token_ids)))
        first_segment_ids = [0]*len(first_token_ids)

        if second_text is not None:
            second_token_ids = self.tokens_to_ids(second_tokens)
            if second_length is not None:
                second_tokens_ids = second_token_ids[:second_length]
                second_token_ids.extend(
                    [self._token_pad_id] *
                    (second_length - len(second_token_ids)))
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        return first_token_ids, first_segment_ids

    def id_to_token(self, i):
        '''
        id序列转换为对应的token
        :param i:
        :return:
        '''
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        '''
        id序列转换为对应的token序列
        :param ids:
        :return:
        '''
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids):
        '''
        转为可读文本
        :param ids:
        :return:
        '''
        raise NotImplementedError

    def truncate_sequence(self,
                          max_length,
                          first_sequence,
                          second_sequence=None,
                          pop_index=1):
        '''
        截断总长度
        :param max_length:
        :param first_sequence:
        :param second_sequence:
        :param pop_index:
        :return:
        '''
        if second_sequence is None:
            second_sequence = []
        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)

    def _tokenize(self, text):
        '''
        基本分词函数
        :param text:
        :return:
        '''
        raise NotImplementedError


class Tokenizer(BasicTokenizer):
    '''
    bert原生分词器
    '''
    def __init__(self, token_dict, do_lower_case=False):
        '''
        初始化
        :param token_dict:
        :param do_lower_case:
        '''
        super(Tokenizer, self).__init__(do_lower_case)
        #token_dict若为string，表示路径
        if is_string(token_dict):
            token_dict = load_vocab(token_dict)
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}

        for token in ['pad','cls','sep','unk','mask']:
            try:
                _token_id =self._token_dict[getattr(self, '_token_%s' % token)]
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass
        self._vocab_size = len(token_dict)
    def token_to_id(self, token):
        '''
        token转换为对应的id
        :param token:
        :return:
        '''
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i):
        '''
        id转换为对应的token
        :param i:
        :return:
        '''
        return self._token_dict_inv[i]

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
               unicodedata.category(ch) == 'Zs'  # Separator, Space

    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
               58 <= code <= 64 or \
               91 <= code <= 96 or \
               123 <= code <= 126 or \
               unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\xb7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
               0x3400 <= code <= 0x4DBF or \
               0x20000 <= code <= 0x2A6DF or \
               0x2A700 <= code <= 0x2B73F or \
               0x2B740 <= code <= 0x2B81F or \
               0x2B820 <= code <= 0x2CEAF or \
               0xF900 <= code <= 0xFAFF or \
               0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def decode(self, ids, tokens=None):
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == "##":
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +', ' ', text)
        text = re.sub(' +', ' ', text)
        #去除空格
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)#去除空格
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        #去除cjk标点空格
        text = re.sub(punctuation_regex, '\\1', text)
        #去除数量之间的空格
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    def _tokenize(self, text):
        '''
        分词函数
        :param text:
        :return:
        '''
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))

        return tokens
    def _word_piece_tokenize(self, word):
        """word内分成subword
        """
        if word in self._token_dict:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens

def load_vocab(dict_path, encoding="utf8", simplified=False, startwith=None):
    '''
    从bert的词典文件中读取词典
    :param dict_path:
    :param encoding:
    :param simplified:
    :param startwith:
    :return:
    '''
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    if simplified:  # 过滤冗余部分token
        new_token_dict, keep_tokens = {}, []
        startwith = startwith or []
        for t in startwith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict:
                keep = True
                if len(t) > 1:
                    for c in (t[2:] if t[:2] == '##' else t):
                        if (Tokenizer._is_cjk_character(c)
                                or Tokenizer._is_punctuation(c)):
                            keep = False
                            break
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict

