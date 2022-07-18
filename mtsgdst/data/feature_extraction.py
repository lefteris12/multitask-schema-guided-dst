import random

import torch

from .. import config
from . import dialogue_processing, utils
from .classes import *


def get_slot_bin_feats(service: Service, slot: Slot, hist: DialogueHistory, turn: Turn):
    current_services = set(turn.frames.keys())
    new_services = current_services - hist.services_so_far
    services_difference = current_services - hist.prev_turn_services
    latest_sys_sv_pairs = hist.services[service.name].latest_sys_sv_pairs
    prev_sys_sv_pairs = hist.services[service.name].prev_sys_sv_pairs

    return {
        'is_new_service': service.name in new_services,
        'service_switched': service.name in services_difference and turn.id != 0,
        'in_sys_uttr': slot.name in latest_sys_sv_pairs and len(latest_sys_sv_pairs[slot.name]) == 1,
        'in_sys_hist': slot.name in prev_sys_sv_pairs and len(prev_sys_sv_pairs[slot.name]) == 1,
        'required_slot': slot.name in service.required_slots,
        'optional_slot': slot.name in service.optional_slots,
    }


def get_bin_feats(service, shuffled_slots, hist, turn):
    binary = torch.empty((config.MAX_SLOTS, config.NUM_BIN_FEATS)).fill_(0)
    for idx, slot_name in enumerate(shuffled_slots):
        slot = service.slots[slot_name]
        binary[idx] = torch.Tensor(list(get_slot_bin_feats(service, slot, hist, turn).values()))
    return binary


def get_intent_labels(service_hist_pred, shuffled_intents, frame):
    num_intents = len(shuffled_intents)
    label = {
        'intent_status': 0,
        'intent_values': torch.empty(config.MAX_INTENTS, dtype=torch.int64).fill_(-1),
    }
    label['intent_values'][:num_intents] = 0

    if frame.state.active_intent != service_hist_pred.intent:
        label['intent_status'] = 1
        intent_idx = shuffled_intents.index(frame.state.active_intent)
        label['intent_values'][intent_idx] = 1

    return label


def get_slot_labels(schema, service, shuffled_slots, hist, hist_pred, frame, sys_usr_uttr, eval_mode):
    label = {
        'usr_status': torch.empty(config.MAX_SLOTS, dtype=torch.int64).fill_(-1),
        'copy_status': torch.empty(config.MAX_SLOTS, dtype=torch.int64).fill_(-1),
        'req_status': torch.empty(config.MAX_SLOTS, dtype=torch.int64).fill_(-1),
        'start_idx': torch.empty(config.MAX_SLOTS, dtype=torch.int64).fill_(-1),
        'end_idx': torch.empty(config.MAX_SLOTS, dtype=torch.int64).fill_(-1),
        'values': torch.empty(config.MAX_VALUES_PER_SERVICE, dtype=torch.int64).fill_(-1),
        'usr_sources': {},
        'sys_sources': {},
    }

    value_indices = []
    shuffled_all_possible_values = []
    updated_sv_pairs = {}
    for idx, slot_name in enumerate(shuffled_slots):
        slot = schema.services[service.name].slots[slot_name]
        usr_status, copy_status, span, values, usr_sources, sys_sources, values_changed = \
            dialogue_processing.parse_service_slot(service, slot, hist, hist_pred, frame)
        if values_changed:
            updated_sv_pairs[slot.name] = values

        if slot_name in schema.services[service.name].usr_slots:
            label['usr_status'][idx] = usr_status
            label['copy_status'][idx] = copy_status
            if not slot.is_categorical:
                slot_desc = ' [SERVICE] ' + schema.services[service.name].pr_name + \
                            ' [SLOT] ' + schema.services[service.name].slots[slot_name].pr_name
                slot_desc = utils.remove_whitespace(slot_desc)
                tokenized = utils.tokenizer(sys_usr_uttr, slot_desc, padding='max_length',
                                            truncation='only_first',
                                            max_length=config.MAX_SEQ_LEN)
                if span:
                    char_start_idx, char_end_idx = span
                    start_idx = tokenized.char_to_token(batch_or_char_index=0, char_index=char_start_idx,
                                                        sequence_index=0)
                    end_idx = tokenized.char_to_token(batch_or_char_index=0, char_index=char_end_idx - 1,
                                                      sequence_index=0)
                    label['start_idx'][idx] = start_idx
                    label['end_idx'][idx] = end_idx
                else:
                    label['start_idx'][idx] = 0
                    label['end_idx'][idx] = 0
            else:
                shuffled_possible_values = slot.possible_values.copy()
                if not eval_mode:
                    random.shuffle(shuffled_possible_values)
                shuffled_all_possible_values += shuffled_possible_values
                possible_values_len = len(shuffled_possible_values)
                curr_slot_values = [0] * possible_values_len
                if usr_status == utils.USR_STATUS_ACTIVE and values[0] != 'dontcare':
                    value_idx = shuffled_possible_values.index(values[0])
                    curr_slot_values[value_idx] = 1

                value_indices += curr_slot_values

        label['values'] = torch.LongTensor(value_indices + (config.MAX_VALUES_PER_SERVICE - len(value_indices)) * [-1])
        label['req_status'][idx] = slot_name in frame.state.req_slots
        label['usr_sources'][slot_name] = usr_sources
        label['sys_sources'][slot_name] = sys_sources
    return label, shuffled_all_possible_values, updated_sv_pairs


def get_cross_copy_labels(service, shuffled_slots, other_service_info, usr_sources, sys_sources):
    cross_copy_labels = torch.empty((config.MAX_SLOTS, config.MAX_SLOTS_OTHER_SERVICE), dtype=torch.long).fill_(-1)
    other_slot_idx = 0
    for other_service_name, other_slot_name in other_service_info:
        if other_service_name == 'None' and other_slot_name == 'None':
            continue
        for target_slot_idx, target_slot_name in enumerate(shuffled_slots):
            if target_slot_name not in service.usr_slots:
                continue
            sources = usr_sources[target_slot_name] + sys_sources[target_slot_name]
            if (other_service_name, other_slot_name) in sources:
                cross_copy_labels[target_slot_idx, other_slot_idx] = 1
            else:
                cross_copy_labels[target_slot_idx, other_slot_idx] = 0
        other_slot_idx += 1
    return cross_copy_labels
