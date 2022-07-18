import os
import re
import json


from .classes import *
from .. import config
from . import utils


def convert_task_name(task_name):
    if task_name == 'single':
        return 'dstc8_single_domain'
    if task_name == 'multi':
        return 'dstc8_multi_domain'
    if task_name == 'all':
        return 'dstc8_all'


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


def load_schema_from_disk(dataset_split):
    full_path = os.path.join(config.DATASET_PATH, dataset_split, 'schema.json')
    with open(full_path, 'rb') as fp:
        schema_json = json.load(fp)

    services = {}
    for service_json in schema_json:
        service_name = service_json['service_name']
        pr_service_name = ' '.join(utils.camel_case_split(service_name)).lower().replace('_', ' ')
        description = service_json['description']
        slots = {}
        intents = {}
        required_slots = set()
        optional_slots = set()
        for intent_json in service_json['intents']:
            intent_name = intent_json['name']
            required_slots.update(intent_json['required_slots'])
            optional_slots.update(intent_json['optional_slots'].keys())
            desc1 = ' '.join(utils.camel_case_split(intent_name))
            desc1 = desc1.lower()
            desc2 = intent_json['description']
            desc = '%s: %s' % (desc1, desc2)
            intents[intent_name] = Intent(
                name=intent_name,
                description=intent_json['description'],
                required_slots=intent_json['required_slots'],
                optional_slots=intent_json['optional_slots'],
                pr_name=desc1,
                pr_name_description=desc
            )

        for slot_json in service_json['slots']:
            slot_name = slot_json['name']
            desc1 = slot_name
            desc1 = desc1.replace('_', ' ')
            desc1 = desc1.lower()
            desc2 = slot_json['description'].lower()
            desc = '%s %s' % (desc1, desc2)
            slots[slot_name] = Slot(
                name=slot_name,
                description=slot_json['description'],
                is_categorical=slot_json['is_categorical'],
                possible_values=slot_json['possible_values'],
                pr_name=desc1,
                pr_name_description=desc
            )

        optional_slots = optional_slots - required_slots
        usr_slots = sorted(required_slots) + sorted(optional_slots)

        services[service_name] = Service(
            name=service_name,
            description=description,
            slots=slots,
            intents=intents,
            required_slots=required_slots,
            optional_slots=optional_slots,
            usr_slots=usr_slots,
            pr_name=pr_service_name
        )

    return Schema(services=services)


def load_dialogues_from_disk(dataset_split):
    dataset_split_path = os.path.join(config.DATASET_PATH, dataset_split)
    dialogues = []
    file_range = get_file_range(dataset_split, config.TASK_NAME)
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
    for dialogue_json in dialogues:
        dial_id = dialogue_json['dialogue_id']
        turns = {}
        for turn_id, turn_json in enumerate(dialogue_json['turns']):
            frames = {}
            for frame_json in turn_json['frames']:
                service_name = frame_json['service']
                slot_spans = {}
                for slot_json in frame_json['slots']:
                    slot_name = slot_json['slot']
                    slot_spans[slot_name] = (slot_json['start'], slot_json['exclusive_end'])
                if 'state' in frame_json:
                    state = State(
                        active_intent=frame_json['state']['active_intent'],
                        req_slots=frame_json['state']['requested_slots'],
                        sv_pairs=frame_json['state']['slot_values'])
                else:
                    state = None

                actions = [Action(name=a['act'], slot=a.get('slot'), values=a.get('values', []))
                           for a in frame_json['actions']]
                frames[service_name] = Frame(
                    actions=actions,
                    state=state,
                    slot_spans=slot_spans
                )
            turns[turn_id] = Turn(
                id=turn_id,
                speaker=turn_json['speaker'],
                utterance=turn_json['utterance'],
                frames=frames
            )
        dialogues_[dial_id] = Dialogue(
            services=dialogue_json['services'],
            turns=turns
        )
    return dialogues_

