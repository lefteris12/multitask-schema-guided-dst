import random
from collections import defaultdict

import torch

from .. import config
from . import utils


def get_input_intents_part(service, shuffled_intents, hist_pred, schema_augment_prob):
    service_hist_pred = hist_pred.services[service.name]
    string = ''
    if config.USE_SERVICE_HISTORY:
        prev_intent = service_hist_pred.intent
        string += ' Active intent '
        string += ' : '
        if prev_intent == 'NONE':
            string += ' [NONE] '
        else:
            string += service.intents[prev_intent].pr_name

    for idx, intent_name in enumerate(shuffled_intents):
        string += ' [INTENT] '
        if intent_name == 'NONE':
            string += ' [NONE] '
        else:
            intent_desc = service.intents[intent_name].pr_name
            if schema_augment_prob > 0:
                prob = random.uniform(0, 1)
                if prob < schema_augment_prob:
                    intent_desc = utils.augment(intent_desc)
            string += intent_desc

    return string


def get_input_slots_part(service, shuffled_slots, shuffled_possible_values, hist_pred, schema_augment_prob):
    service_hist_pred = hist_pred.services[service.name]

    value_idx = 0
    string = ''

    for idx, slot_name in enumerate(shuffled_slots):
        string += ' [SLOT] '

        slot_desc = service.slots[slot_name].pr_name
        if slot_name in service.usr_slots and config.USE_FULL_SLOT_DESCRIPTIONS:
            slot_desc = service.slots[slot_name].pr_name_description

        if schema_augment_prob > 0:
            prob = random.uniform(0, 1)
            if prob < schema_augment_prob:
                slot_desc = utils.augment(slot_desc)

        string += slot_desc

        if slot_name not in service.usr_slots:
            continue

        if config.USE_SERVICE_HISTORY:
            string += ' : '
            if slot_name in service_hist_pred.usr_sv_pairs and service_hist_pred.usr_sv_pairs[slot_name]:
                string += service_hist_pred.usr_sv_pairs[slot_name][0]
            else:
                string += ' [NONE] '

        slot = service.slots[slot_name]
        if slot.is_categorical:
            possible_values_len = len(slot.possible_values)
            possible_values = shuffled_possible_values[value_idx:value_idx + possible_values_len]
            value_idx += possible_values_len
            for value in possible_values:
                value_desc = value
                if schema_augment_prob > 0 and not value.isnumeric():
                    prob = random.uniform(0, 1)
                    if prob < schema_augment_prob:
                        value_desc = utils.augment(value)
                string += ' [VALUE] ' + value_desc

    return string


def can_slot_cross_copied(s):
    if not s.is_categorical:
        return True
    if s.possible_values[0].isnumeric():
        return True
    return False


def get_input_other_service_slots_part(schema, service, hist, hist_pred):
    other_service_info = []

    usr_slots_with_values = defaultdict(dict)
    string = ''

    for other_ser_name, other_ser in hist.services.items():
        if other_ser_name == service.name:
            continue
        for other_slot_name, other_slot_values in other_ser.sys_sv_pairs.items():
            other_slot = schema.services[other_ser_name].slots[other_slot_name]
            if other_slot_values and can_slot_cross_copied(other_slot):
                usr_slots_with_values[other_ser_name][other_slot_name] = (other_slot_values[0], False)

    for other_ser_name, other_ser in hist_pred.services.items():
        if other_ser_name == service.name:
            continue
        for other_slot_name, other_slot_values in other_ser.usr_sv_pairs.items():
            other_slot = schema.services[other_ser_name].slots[other_slot_name]
            if other_slot_values and can_slot_cross_copied(other_slot):
                usr_slots_with_values[other_ser_name][other_slot_name] = (other_slot_values[0], True)

    shuffled_keys = list(usr_slots_with_values.keys())
    random.shuffle(shuffled_keys)
    for other_service_name in shuffled_keys:
        other_service = usr_slots_with_values[other_service_name]
        string += ' [SERVICE] '
        string += schema.services[other_service_name].pr_name
        shuffled_keys2 = list(other_service.keys())
        random.shuffle(shuffled_keys2)

        for other_slot_name in shuffled_keys2:
            other_slot_value, is_usr_slot = other_service[other_slot_name]
            string += ' [SLOT] '
            if not is_usr_slot:
                string += ' system '
            string += schema.services[other_service_name].slots[other_slot_name].pr_name
            if config.USE_DIALOGUE_HISTORY:
                string += ' : '
                string += str(other_slot_value)

            other_service_info.append((other_service_name, other_slot_name))

    # Padding
    other_service_info += [('None', 'None')] * (config.MAX_SLOTS_OTHER_SERVICE - len(other_service_info))

    return string, other_service_info


def construct_input_sequence(schema, service, shuffled_intents, shuffled_slots, shuffled_possible_values, hist,
                             hist_pred, sys_usr_uttr, schema_augment_prob):
    # Part 1 & 2: System and user utterances
    uttr = sys_usr_uttr

    # PART 3: Intents
    schema_desc = '[SERVICE] ' + service.pr_name
    schema_desc += get_input_intents_part(service, shuffled_intents, hist_pred, schema_augment_prob)

    # PART 4: Slots
    schema_desc += get_input_slots_part(
        service, shuffled_slots, shuffled_possible_values, hist_pred, schema_augment_prob)

    # Part 5: Other service slots
    string, other_service_info = get_input_other_service_slots_part(schema, service, hist, hist_pred)
    schema_desc += string

    schema_desc = utils.remove_whitespace(schema_desc)
    tokenized = utils.tokenizer(uttr, schema_desc, padding='max_length',
                                truncation='only_first',
                                max_length=config.MAX_SEQ_LEN)
    return tokenized, other_service_info


# Return a mask of the user utterance tokens to perform span classification
def get_usr_uttr_mask(tokenized):
    # First [SEP] of the first segment
    start = tokenized['input_ids'].index(utils.tokenizer.sep_token_id) + 1
    end = tokenized['token_type_ids'].index(1) - 2
    usr_uttr_mask = torch.zeros(len(tokenized['input_ids']))
    usr_uttr_mask[start:end + 1] = 1
    return usr_uttr_mask


# Return the positions of the special tokens ([INTENT], [SLOT], [VALUE]) in the input sequence
def get_special_token_positions(tokenized):
    slot_positions = []
    slot_positions_other_service = []
    intent_positions = []
    value_positions = []
    service_tokens_sofar = 0
    for idx, (input_id, type_id) in enumerate(zip(tokenized['input_ids'], tokenized['token_type_ids'])):
        if type_id == 0:
            continue
        if input_id == utils.special_token_name_to_id['[SERVICE]']:
            service_tokens_sofar += 1
        if input_id == utils.special_token_name_to_id['[SLOT]']:
            if service_tokens_sofar == 1:
                slot_positions.append(idx)
            else:
                slot_positions_other_service.append(idx)
        elif input_id == utils.special_token_name_to_id['[INTENT]']:
            intent_positions.append(idx)
        elif input_id == utils.special_token_name_to_id['[VALUE]'] and service_tokens_sofar == 1:
            value_positions.append(idx)

    slot_positions += [0] * (config.MAX_SLOTS - len(slot_positions))
    slot_positions_other_service += [0] * (config.MAX_SLOTS_OTHER_SERVICE - len(slot_positions_other_service))
    intent_positions += [0] * (config.MAX_INTENTS - len(intent_positions))
    value_positions += [0] * (config.MAX_VALUES_PER_SERVICE - len(value_positions))

    return {
        'slot_positions': torch.LongTensor(slot_positions),
        'slot_positions_other_service': torch.LongTensor(slot_positions_other_service),
        'intent_positions': torch.LongTensor(intent_positions),
        'value_positions': torch.LongTensor(value_positions),
    }
