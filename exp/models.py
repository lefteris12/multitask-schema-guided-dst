import torch
from torch import nn

from exp import data_utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ClassificationHead(nn.Module):
    def __init__(self, dropout, input_dim, hidden_dim, output_dim, num_bin_feats=0):
        super(ClassificationHead, self).__init__()
        self.num_bin_feats = num_bin_feats

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim + num_bin_feats, output_dim)

    def forward(self, x):
        out = self.linear1(x['cls'])
        out = torch.relu(out)
        out = self.dropout(out)
        if self.num_bin_feats > 0:
            out = torch.cat([out, x['binary']], dim=-1)
        logits = self.linear2(out)
        return logits


class AllSlotsModel(nn.Module):
    def __init__(self, dropout, use_binary_feats=True):
        super(AllSlotsModel, self).__init__()
        self.bert = data_utils.get_bert_with_tokens()
        self.dropout1 = nn.Dropout(dropout)

        # Categorical slots
        self.cl_value = ClassificationHead(dropout, self.bert.config.hidden_size, self.bert.config.hidden_size, 2)
        # Non-categorical slots
        self.cl_start = ClassificationHead(dropout, 2 * self.bert.config.hidden_size, self.bert.config.hidden_size, 1)
        self.cl_end = ClassificationHead(dropout, 2 * self.bert.config.hidden_size, self.bert.config.hidden_size, 1)
        # Cross
        self.cl_cross = ClassificationHead(dropout, 2 * self.bert.config.hidden_size, self.bert.config.hidden_size, 2)

        self.cl_intent_status = ClassificationHead(dropout, self.bert.config.hidden_size, 128, 2)
        self.cl_intent_value = ClassificationHead(dropout, self.bert.config.hidden_size, 128, 2)
        if use_binary_feats:
            self.cl_usr_status = ClassificationHead(dropout, self.bert.config.hidden_size, 16, 3, data_utils.NUM_BIN_FEATS)
            self.cl_copy_status = ClassificationHead(dropout, self.bert.config.hidden_size, 16, 4, data_utils.NUM_BIN_FEATS)
            self.cl_req_status = ClassificationHead(dropout, self.bert.config.hidden_size, 16, 2, data_utils.NUM_BIN_FEATS)
        else:
            self.cl_usr_status = ClassificationHead(dropout, self.bert.config.hidden_size, 16, 3)
            self.cl_copy_status = ClassificationHead(dropout, self.bert.config.hidden_size, 16, 4)
            self.cl_req_status = ClassificationHead(dropout, self.bert.config.hidden_size, 16, 2)

    def forward(self, x):
        logits = {}
        # y: (batch_size, seq_len, self.bert.config.hidden_size)
        # x['slot_positions']: (batch_size, MAX_SLOTS)
        # x['value_positions']: (batch_size, MAX_VALUES_PER_SERVICE)
        # slot_embeddings: (batch_size, MAX_SLOTS, self.bert.config.hidden_size)
        # value_embeddings: (batch_size, MAX_VALUES_PER_SERVICE, self.bert.config.hidden_size)
        y = self.bert(**x['tokenized']).last_hidden_state
        y = self.dropout1(y)
        slot_embeddings = torch.gather(y, 1, x['slot_positions'].unsqueeze(-1).repeat(1, 1, self.bert.config.hidden_size))
        intent_embeddings = torch.gather(y, 1, x['intent_positions'].unsqueeze(-1).repeat(1, 1, self.bert.config.hidden_size))
        slot_embeddings_other = torch.gather(y, 1, x['slot_positions_other_service'].unsqueeze(-1).repeat(1, 1, self.bert.config.hidden_size))
        value_embeddings = torch.gather(y, 1, x['value_positions'].unsqueeze(-1).repeat(1, 1, self.bert.config.hidden_size))

        logits['intent_status'] = self.cl_intent_status({'cls': y[:, 0, :]})
        logits['intent_values'] = self.cl_intent_value({'cls': intent_embeddings})

        slot_feats = {'cls': slot_embeddings, 'binary': x['binary']}
        logits['usr_status'] = self.cl_usr_status(slot_feats)
        logits['copy_status'] = self.cl_copy_status(slot_feats)
        logits['req_status'] = self.cl_req_status(slot_feats)
        logits['values'] = self.cl_value({'cls': value_embeddings})

        slot_embeddings_repeated = slot_embeddings.unsqueeze(2).repeat(1, 1, y.shape[1], 1)
        tokens_repeated = y.unsqueeze(1).repeat(1, slot_embeddings.shape[1], 1, 1)
        slots_and_tokens = torch.cat([slot_embeddings_repeated, tokens_repeated], 3)

        logits['start'] = self.cl_start({'cls': slots_and_tokens}).squeeze(-1)
        logits['end'] = self.cl_end({'cls': slots_and_tokens}).squeeze(-1)

        usr_uttr_mask = x['usr_uttr_mask']
        usr_uttr_mask[:, 0] = 1
        usr_uttr_mask = usr_uttr_mask.unsqueeze(1)

        logits['start'] = torch.where(usr_uttr_mask == 1, logits['start'], torch.Tensor([-1e9]).to(device))
        logits['end'] = torch.where(usr_uttr_mask == 1, logits['end'], torch.Tensor([-1e9]).to(device))

        # slot_embeddings: (batch_size, MAX_SLOTS, self.bert.config.hidden_size)
        # slot_embeddings_other: (batch_size, MAX_SLOTS_OTHER, self.bert.config.hidden_size)
        # slots_cross: (batch_size, MAX_SLOTS, MAX_SLOTS_OTHER, 2 * self.bert.config.hidden_size)
        slot_embeddings_repeated2 = slot_embeddings.unsqueeze(2).repeat(1, 1, data_utils.MAX_SLOTS_OTHER_SERVICE, 1)
        slot_embeddings_other_repeated2 = slot_embeddings_other.unsqueeze(1).repeat(1, data_utils.MAX_SLOTS, 1, 1)
        slots_cross = torch.cat([slot_embeddings_repeated2, slot_embeddings_other_repeated2], dim=-1)
        logits['cross'] = self.cl_cross({'cls': slots_cross})

        return logits
