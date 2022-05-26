import copy
import pickle
import re
from collections import defaultdict
import os
import json
import argparse
from transformers import AutoTokenizer, BertModel
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import random

from exp.eda import *


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dataset_path = 'dstc8-schema-guided-dialogue'

USR_STATUS_NONE = 0
USR_STATUS_ACTIVE = 1
USR_STATUS_DONTCARE = 2

COPY_STATUS_NONE = 0
COPY_STATUS_SYS_UTTR = 1
COPY_STATUS_SYS_HIST = 2
COPY_STATUS_CROSS = 3

MAX_INTENTS = 5
MAX_CAT_VALUES = 11
MAX_SLOTS = 17
MAX_SLOTS_OTHER_SERVICE = 40
MAX_VALUES_PER_SERVICE = 23
NUM_BIN_FEATS = 6

with open('exp/model_name', 'r') as f:
    MODEL_NAME = f.read().strip()

special_tokens_list = ['[SERVICE]', '[ACTION]', '[SLOT]', '[VALUE]', '[INTENT]', '[NONE]']
special_tokens_dict = {'additional_special_tokens': special_tokens_list}
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
special_token_name_to_id = {name: id_ for name, id_ in zip(special_tokens_list, tokenizer.additional_special_tokens_ids)}


## HELPERS
def remove_whitespace(x):
    x = x.strip()
    x = ' '.join(x.split())
    return x


def get_bert_with_tokens():
    # https://github.com/huggingface/tokenizers/issues/247
    model = BertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    return model


def camel_case_split(x):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', x)
    return [m.group(0) for m in matches]


def iterate_dict(func, x):
    if isinstance(x, dict):
        for y in x.values():
            iterate_dict(func, y)
    else:
        func(x)


def convert_task_name(task_name):
    if task_name == 'single':
        return 'dstc8_single_domain'
    if task_name == 'multi':
        return 'dstc8_multi_domain'
    if task_name == 'all':
        return 'dstc8_all'


def compatible_slots(slot_x, slot_y):
    if slot_x['is_categorical'] != slot_y['is_categorical']:
        return False

    def numerical(slot):
        return slot['possible_values'][0].isnumeric()

    if slot_x['is_categorical']:
        if not (numerical(slot_x) and numerical(slot_y)):
            return False

    return True


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


## SCHEMA

# Parse the schema
def load_schema(dataset_split):
    full_path = os.path.join(dataset_path, dataset_split, 'schema.json')
    with open(full_path, 'rb') as fp:
        schema = json.load(fp)

    schema_ = {}
    for service in schema:
        service_name = service['service_name']
        schema_[service_name] = {}
        schema_[service_name]['slots'] = {}
        schema_[service_name]['intents'] = {}
        schema_[service_name]['description'] = service['description']
        for slot in service['slots']:
            slot_name = slot['name']
            schema_[service_name]['slots'][slot_name] = slot

        # Slots that the user is allowed to give a value
        schema_[service_name]['required_slots'] = set()
        schema_[service_name]['optional_slots'] = set()
        for intent in service['intents']:
            intent_name = intent['name']
            schema_[service_name]['intents'][intent_name] = intent

            schema_[service_name]['required_slots'].update(intent['required_slots'])
            schema_[service_name]['optional_slots'].update(intent['optional_slots'].keys())

        schema_[service_name]['optional_slots'] = schema_[service_name]['optional_slots'] - schema_[service_name]['required_slots']
        schema_[service_name]['usr_slots'] = sorted(schema_[service_name]['required_slots']) + sorted(schema_[service_name]['optional_slots'])
    return schema_


# Load the schema element names and descriptions
def create_schema_features(schema):
    features = {
        'intents': defaultdict(dict),
        'slots': defaultdict(dict),
        'intent_names': defaultdict(dict),
        'slot_names': defaultdict(dict),
        'service_names': {},
    }
    desc = '%s: %s' % ('None', 'No active intent')
    features['intent_none'] = desc

    for service_name, service in schema.items():
        features['service_names'][service_name] = ' '.join(camel_case_split(service_name)).lower().replace('_', ' ')
        for intent_name, intent in service['intents'].items():
            desc1 = ' '.join(camel_case_split(intent_name))
            desc1 = desc1.lower()
            desc2 = intent['description']
            desc = '%s: %s' % (desc1, desc2)
            features['intents'][service_name][intent_name] = desc
            features['intent_names'][service_name][intent_name] = desc1

        for slot_name, slot in service['slots'].items():
            desc1 = slot_name
            desc1 = desc1.replace('_', ' ')
            desc1 = desc1.lower()
            desc2 = slot['description'].lower()
            desc = '%s %s' % (desc1, desc2)
            features['slots'][service_name][slot_name] = desc
            features['slot_names'][service_name][slot_name] = desc1

    return features


## DIALOGUE
def get_file_range(dataset_split, task_name):
    if isinstance(task_name, range):
        return task_name

    if dataset_split == 'train':
        if task_name == 'debug':
            return range(1, 2)
        if task_name == 'single':
            return range(1, 44)
        elif task_name == 'multi':
            return range(44, 128)
        elif task_name == 'all':
            return range(1, 128)

        else:
            raise ValueError('Wrong value for task_name.')
    elif dataset_split == 'dev':
        if task_name == 'single':
            return range(1, 8)
        elif task_name == 'multi':
            return range(8, 21)
        elif task_name == 'all':
            return range(1, 21)
        else:
            raise ValueError('Wrong value for task_name.')
    elif dataset_split == 'test':
        if task_name == 'single':
            return range(1, 12)
        elif task_name == 'multi':
            return range(12, 35)
        elif task_name == 'all':
            return range(1, 35)
        else:
            raise ValueError('Wrong value for task_name.')
    else:
        raise ValueError('Wrong value for dataset_split.')


# Parse the dialogues
def load_dialogues(dataset_split, task_name):
    dataset_split_path = os.path.join(dataset_path, dataset_split)
    dialogues = []
    file_range = get_file_range(dataset_split, task_name)
    for file_name in sorted(os.listdir(dataset_split_path)):
        full_path = os.path.join(dataset_split_path, file_name)
        m = re.match(r'dialogues_(...).json', file_name)
        if not m:
            continue
        integer = int(m.group(1))
        if integer not in file_range:
            continue

        print(f'Loading {full_path}')
        with open(full_path, 'rb') as fp:
            dialogues += json.load(fp)

    dialogues_ = {}
    for dialogue in dialogues:
        dial_id = dialogue['dialogue_id']
        dialogues_[dial_id] = {
            'services': dialogue['services'],
            'turns': {},
        }
        for turn_id, turn in enumerate(dialogue['turns']):
            dialogues_[dial_id]['turns'][turn_id] = {
                'speaker': turn['speaker'],
                'utterance': turn['utterance'],
                'frames': {},
            }
            for frame in turn['frames']:
                service_name = frame['service']
                dialogues_[dial_id]['turns'][turn_id]['frames'][service_name] = {
                    'slots': {},
                    'actions': frame['actions'],
                    'state': {},
                }
                for slot in frame['slots']:
                    slot_name = slot['slot']
                    dialogues_[dial_id]['turns'][turn_id]['frames'][service_name]['slots'][slot_name] = {
                        'start': slot['start'],
                        'exclusive_end': slot['exclusive_end'],
                    }
                if 'state' in frame:
                    dialogues_[dial_id]['turns'][turn_id]['frames'][service_name]['state'] = frame['state']
    return dialogues_


# The main function that processes the dialogues
def create_dialogues_dict(dialogues, schema, schema_features, use_natural_sys_uttr=False):
    data = defaultdict(lambda: defaultdict(dict))
    items = dialogues.items()

    for dial_id, dial in items:
        is_multi_domain = len(dial['services']) > 1
        prev_sys_uttr = None
        prev_active_intent = {}
        sys_history = {}
        prev_sys_history = {}
        turn_sys_history = {}
        usr_history = {}
        services_so_far = set()
        prev_services = set()

        for service_name in dial['services']:
            sys_history[service_name] = {}
            prev_sys_history[service_name] = {}
            turn_sys_history[service_name] = {}
            usr_history[service_name] = {}
            prev_active_intent[service_name] = 'NONE'

        for turn_id, turn in dial['turns'].items():
            current_services = set(turn['frames'].keys())
            new_services = current_services - services_so_far
            services_so_far.update(turn['frames'].keys())

            if turn['speaker'] == 'SYSTEM':
                prev_sys_uttr = ''
                natural_sys_uttr = turn['utterance']

                turn_sys_history = {}
                for service_name in dial['services']:
                    turn_sys_history[service_name] = {}

                assert len(turn['frames']) == 1
                for service_name, frame in turn['frames'].items():
                    prev_sys_uttr += ' [SERVICE] ' + schema_features['service_names'][service_name]
                    for action in frame['actions']:
                        prev_sys_uttr += ' [ACTION] '
                        # Name of the act
                        prev_sys_uttr += action['act'].capitalize().replace('_', ' ')
                        # Act slot
                        if 'slot' in action:
                            if action['act'] not in ('OFFER_INTENT', 'INFORM_COUNT'):
                                prev_sys_uttr += ' [SLOT] '
                                prev_sys_uttr += action['slot'].lower().replace('_', ' ')
                            # Act values
                            if len(action['values']) > 0:
                                prev_sys_uttr += ' [VALUE] '
                                if action['act'] == 'OFFER_INTENT':
                                    values = schema_features['intent_names'][service_name][action['values'][0]]
                                else:
                                    values = ' [VALUE] '.join(action['values'])
                                prev_sys_uttr += values

                        if action['act'] in ('INFORM', 'CONFIRM', 'OFFER') or \
                                action['act'] == 'REQUEST' and len(action['values']) == 1:
                            turn_sys_history[service_name][action['slot']] = (action['values'], action['act'])
                            sys_history[service_name][action['slot']] = (action['values'], action['act'])

                prev_sys_uttr = prev_sys_uttr.strip()
                prev_sys_uttr = ' '.join(prev_sys_uttr.split())
                prev_sys_uttr = remove_whitespace(prev_sys_uttr)

                if use_natural_sys_uttr:
                    prev_sys_uttr = natural_sys_uttr
            elif turn['speaker'] == 'USER':
                services_difference = current_services - prev_services
                prev_services = current_services

                sys_uttr = prev_sys_uttr if prev_sys_uttr else ''
                usr_uttr = turn['utterance']

                sys_usr_uttr = "%s [SEP] %s" % (sys_uttr, usr_uttr)
                data[dial_id][turn_id] = {
                    'sys_usr_uttr': sys_usr_uttr,
                    'usr_uttr': usr_uttr,
                    'curr_sys_history': copy.deepcopy(sys_history),
                    'usr_history_sofar': copy.deepcopy(usr_history),
                    'intent_history_sofar': copy.deepcopy(prev_active_intent),
                    'services': {},
                }

                for service_name, frame in turn['frames'].items():
                    service_switched = service_name in services_difference and turn_id != 0
                    is_new_service = service_name in new_services
                    active_intent = frame['state']['active_intent']

                    data[dial_id][turn_id]['services'][service_name] = {
                        'active_intent': active_intent,
                        'prev_active_intent': prev_active_intent[service_name],
                        'req_slots': frame['state']['requested_slots'],
                        'slots': {},
                        'sys_history': copy.deepcopy(prev_sys_history),
                    }
                    prev_active_intent[service_name] = active_intent
                    for slot_name in schema[service_name]['slots']:
                        slot_span, usr_status, values = None, None, None
                        copy_status = COPY_STATUS_NONE
                        usr_sources, sys_sources = [], []
                        prev_hist = prev_sys_history[service_name]
                        curr_hist = turn_sys_history[service_name]
                        is_categorical = schema[service_name]['slots'][slot_name]['is_categorical']

                        if slot_name in usr_history[service_name]:
                            prev_values = usr_history[service_name][slot_name][0]
                        else:
                            prev_values = []

                        if slot_name in sys_history[service_name] and len(sys_history[service_name][slot_name][0]) == 1:
                            hist_value = sys_history[service_name][slot_name][0][0]
                        else:
                            hist_value = None

                        if slot_name in frame['state']['slot_values']:
                            values = frame['state']['slot_values'][slot_name]
                            values_changed = (set(prev_values) != set(values)) or (slot_name in frame['slots'])

                            in_sys_hist = slot_name in prev_hist and len(prev_hist[slot_name][0]) == 1 and \
                                          values and prev_hist[slot_name][0][0] in values
                            in_sys_uttr = slot_name in curr_hist and len(curr_hist[slot_name][0]) == 1 and \
                                          values and curr_hist[slot_name][0][0] in values

                            if values_changed:
                                if values == ['dontcare']:
                                    usr_status = USR_STATUS_DONTCARE
                                else:
                                    if not is_categorical and slot_name in frame['slots']:
                                        orig_start = frame['slots'][slot_name]['start']
                                        orig_end = frame['slots'][slot_name]['exclusive_end']
                                        offset = len(sys_uttr) + len(' [SEP] ')
                                        slot_span = (orig_start + offset, orig_end + offset)
                                        usr_status = USR_STATUS_ACTIVE

                                    if usr_status != USR_STATUS_ACTIVE:
                                        is_slot_informed = False
                                        for action in frame['actions']:
                                            if action['act'] == 'INFORM' and action['slot'] == slot_name:
                                                is_slot_informed = True

                                        if is_categorical:
                                            if is_slot_informed:
                                                usr_status = USR_STATUS_ACTIVE
                                            elif in_sys_uttr:
                                                usr_status = USR_STATUS_NONE
                                                copy_status = COPY_STATUS_SYS_UTTR
                                            elif in_sys_hist and not in_sys_uttr:
                                                usr_status = USR_STATUS_NONE
                                                copy_status = COPY_STATUS_SYS_HIST
                                            elif is_multi_domain:
                                                usr_status = USR_STATUS_NONE
                                                copy_status = COPY_STATUS_CROSS
                                            else:
                                                assert False
                                        else:
                                            assert not is_slot_informed
                                            if in_sys_uttr:
                                                prev_sys_frame = dial['turns'][turn_id-1]['frames'][service_name]
                                                orig_start = prev_sys_frame['slots'][slot_name]['start']
                                                orig_end = prev_sys_frame['slots'][slot_name]['exclusive_end']
                                                # slot_span = (orig_start, orig_end)
                                                usr_status = USR_STATUS_NONE
                                                copy_status = COPY_STATUS_SYS_UTTR
                                            elif in_sys_hist and not in_sys_uttr:
                                                usr_status = USR_STATUS_NONE
                                                copy_status = COPY_STATUS_SYS_HIST
                                            elif is_multi_domain:
                                                usr_status = USR_STATUS_NONE
                                                if len(set(prev_values) & set(values)) == 0:
                                                    copy_status = COPY_STATUS_CROSS
                                            else:
                                                usr_status = USR_STATUS_NONE

                                        if copy_status == COPY_STATUS_CROSS:
                                            for other_service in usr_history:
                                                if other_service != service_name:
                                                    for other_slot in usr_history[other_service]:
                                                        other = usr_history[other_service][other_slot][0]
                                                        if other and (set(other) & set(values)):
                                                            usr_sources.append((other_service, other_slot))

                                            for other_service in sys_history:
                                                if other_service != service_name:
                                                    for other_slot in sys_history[other_service]:
                                                        other = sys_history[other_service][other_slot][0]
                                                        if other and (set(other) & set(values)):
                                                            sys_sources.append((other_service, other_slot))

                                usr_history[service_name][slot_name] = (values, turn_id)
                            else:
                                # Values have not changed
                                usr_status = USR_STATUS_NONE
                        else:
                            # Slot name not in user state frame
                            usr_status = USR_STATUS_NONE

                        if slot_name in curr_hist and len(curr_hist[slot_name][0]) == 1 and values and curr_hist[slot_name][0][0] in values:
                            copy_status = COPY_STATUS_SYS_UTTR

                        candidate_sources, candidate_sources_with_values = [], []
                        for other_service_name in set(services_so_far):
                            if other_service_name == service_name:
                                continue

                            for other_slot_name in schema[other_service_name]['slots']:
                                slot1 = schema[service_name]['slots'][slot_name]
                                slot2 = schema[other_service_name]['slots'][other_slot_name]
                                if compatible_slots(slot1, slot2):
                                    candidate_sources.append((other_service_name, other_slot_name))
                                    if other_slot_name in usr_history[other_service_name] or other_slot_name in sys_history[other_service_name]:
                                        candidate_sources_with_values.append((other_service_name, other_slot_name))

                        if copy_status == COPY_STATUS_CROSS:
                            assert set(usr_sources + sys_sources).issubset(candidate_sources_with_values)

                        in_sys_uttr = slot_name in curr_hist and len(curr_hist[slot_name][0]) == 1
                        in_sys_hist = slot_name in prev_hist and len(prev_hist[slot_name][0]) == 1
                        required_slot = slot_name in schema[service_name]['required_slots']
                        optional_slot = slot_name in schema[service_name]['optional_slots']

                        data[dial_id][turn_id]['services'][service_name]['slots'][slot_name] = {
                            'hist_value': hist_value,
                            'candidate_sources': candidate_sources,
                            'is_new_service': is_new_service,
                            'service_switched': service_switched,
                            'in_sys_uttr': in_sys_uttr,
                            'in_sys_hist': in_sys_hist,
                            'required_slot': required_slot,
                            'optional_slot': optional_slot,
                            # Only for training
                            'usr_status': usr_status,
                            'copy_status': copy_status,
                            'span': slot_span,
                            'values': values,
                            'usr_sources': usr_sources,
                            'sys_sources': sys_sources,
                            'candidate_sources_with_values': candidate_sources_with_values,
                        }

                prev_sys_history = copy.deepcopy(sys_history)

    data = dict(data)
    data = {k: dict(v) for k, v in data.items()}
    return data
