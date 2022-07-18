from pathlib import Path

from transformers import AutoTokenizer, BertModel

from .. import config
from .eda import *

USR_STATUS_NONE = 0
USR_STATUS_ACTIVE = 1
USR_STATUS_DONTCARE = 2

COPY_STATUS_NONE = 0
COPY_STATUS_SYS_UTTR = 1
COPY_STATUS_SYS_HIST = 2
COPY_STATUS_CROSS = 3


special_tokens_list = ['[SERVICE]', '[ACTION]', '[SLOT]', '[VALUE]', '[INTENT]', '[NONE]']
special_tokens_dict = {'additional_special_tokens': special_tokens_list}
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
special_token_name_to_id = {name: id_ for name, id_ in zip(special_tokens_list, tokenizer.additional_special_tokens_ids)}


def camel_case_split(x):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', x)
    return [m.group(0) for m in matches]


def remove_whitespace(x):
    x = x.strip()
    x = ' '.join(x.split())
    return x


def get_bert_with_tokens():
    # https://github.com/huggingface/tokenizers/issues/247
    model = BertModel.from_pretrained(config.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    return model


def augment(x):
    num_aug = 1
    alpha_sr = 0.1
    alpha_ri = 0.0
    alpha_rs = 0.1
    alpha_rd = 0.0
    try:
        aug_sentences = eda(x, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd,
                            num_aug=num_aug)
        return aug_sentences[0]
    except:
        return x


# Randomly remove tokens from the user utterance
def apply_word_dropout(tokenized, usr_uttr_mask, dropout_rate):
    if dropout_rate > 0:
        for idx, input_id in enumerate(tokenized['input_ids']):
            prob = random.uniform(0, 1)
            if usr_uttr_mask[idx] == 1 and prob < dropout_rate:
                tokenized['input_ids'][idx] = tokenizer.unk_token_id
    return tokenized


def move_to_gpu(arg):
    if isinstance(arg, dict):
        return {k: move_to_gpu(v) for k, v in arg.items()}
    try:
        return arg.to(config.DEVICE)
    except:
        return arg


def create_directories():
    for dir in ('pickles', 'checkpoints', 'temp', 'figs'):
        Path(dir).mkdir(parents=True, exist_ok=True)
