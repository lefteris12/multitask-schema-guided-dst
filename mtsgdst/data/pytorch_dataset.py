import argparse
import copy
import pickle
import random

from torch.utils.data import Dataset, default_collate
from tqdm.auto import tqdm

from .. import config
from . import input_sequence, dialogue_processing, feature_extraction, utils, sgd_dataset
from .classes import *


# This custom collate_fn reduces the input sequence size to the longest input sequence in the batch
def collate_fn(data):
    min_padded = 9999999
    for data_ in data:
        tokenized = data_[0]['tokenized']
        input_ids = list(tokenized['input_ids'])
        padded = input_ids.count(utils.tokenizer.pad_token_id)
        min_padded = min(padded, min_padded)
        max_length = len(input_ids)

    new_max_length = max_length - min_padded
    new_data = []
    for data_ in data:
        tokenized = data_[0]['tokenized']
        tokenized['input_ids'] = tokenized['input_ids'][:new_max_length]
        tokenized['token_type_ids'] = tokenized['token_type_ids'][:new_max_length]
        tokenized['attention_mask'] = tokenized['attention_mask'][:new_max_length]
        new_data_ = data_
        new_data_[0]['usr_uttr_mask'] = new_data_[0]['usr_uttr_mask'][:new_max_length]
        data_[0]['tokenized'] = tokenized
        new_data.append(new_data_)

    return default_collate(new_data)


class MultitaskDataset(Dataset):
    def __init__(self, dialogues, schema, eval_mode, augment):
        self.eval_mode = eval_mode
        self.word_dropout = config.WORD_DROPOUT if augment else 0
        self.schema_augment_prob = config.SCHEMA_AUGMENT_PROB if augment else 0
        self.dialogues = dialogues
        self.schema = schema

        if self.eval_mode:
            self.running_hist_pred = {}

        self.examples, self.histories, self.histories_pred = self.parse_dialogues()
        print(f'Dataset with {len(self.examples)} examples')

    def parse_dialogues(self):
        examples = []
        histories = {}
        histories_pred = {}
        for dial_id, d in tqdm(self.dialogues.items()):
            # Initialize history for all services found in the dialogue
            hist = DialogueHistory()
            for service_name in d.services:
                hist.services[service_name] = ServiceHistory()

            hist_pred = DialogueHistoryPred()
            for service_name in d.services:
                hist_pred.services[service_name] = ServiceHistoryPred()

            # Only initialize running history at the start of the dialogue
            # because it must be updated according to the model outputs during evaluation
            if self.eval_mode:
                self.running_hist_pred[dial_id] = copy.deepcopy(hist_pred)

            for turn_id, turn in d.turns.items():
                if turn.speaker == 'SYSTEM':
                    sys_uttr, sv_pairs = dialogue_processing.parse_system_turn(self.schema, turn)
                    dialogue_processing.get_new_hist_after_system_turn(hist, sys_uttr, sv_pairs)
                    continue

                histories[(dial_id, turn_id)] = hist
                if not self.eval_mode:
                    histories_pred[(dial_id, turn_id)] = hist_pred

                new_hist = copy.deepcopy(hist)
                new_hist_pred = copy.deepcopy(hist_pred)
                for service_name, frame in turn.frames.items():
                    service = self.schema.services[service_name]
                    shuffled_slots = list(service.slots.keys()).copy()
                    shuffled_intents = list(service.intents.keys()).copy()
                    shuffled_intents += ['NONE']

                    if not self.eval_mode:
                        random.shuffle(shuffled_slots)
                        random.shuffle(shuffled_intents)

                    sys_uttr = hist.prev_sys_uttr
                    usr_uttr = turn.utterance
                    sys_usr_uttr = "%s [SEP] %s" % (sys_uttr, usr_uttr)
                    service_hist_pred = hist_pred.services[service_name]
                    intent_labels = feature_extraction.get_intent_labels(service_hist_pred, shuffled_intents, frame)
                    slot_labels, shuffled_all_possible_values, updated_sv_pairs = \
                        feature_extraction.get_slot_labels(self.schema, service, shuffled_slots, hist, hist_pred, frame,
                                                           sys_usr_uttr, self.eval_mode)
                    binary_feats = feature_extraction.get_bin_feats(service, shuffled_slots, hist, turn)

                    dialogue_processing.get_new_service_hist_after_user_turn(
                        new_hist_pred.services[service_name], frame.state.active_intent, updated_sv_pairs)

                    example = {
                        'dial_id': dial_id,
                        'turn_id': turn_id,
                        'service_name': service_name,
                        'binary': binary_feats,
                        'shuffled_slots': shuffled_slots,
                        'shuffled_intents': shuffled_intents,
                        'shuffled_possible_values': shuffled_all_possible_values,
                        'uttr': sys_usr_uttr,
                    }
                    labels = {**intent_labels, **slot_labels}

                    examples.append((example, labels))

                dialogue_processing.get_new_hist_after_user_turn(new_hist, turn)

                hist = new_hist
                hist_pred = new_hist_pred

        return examples, histories, histories_pred

    def __getitem__(self, idx):
        example, label = self.examples[idx]
        dial_id, turn_id, service_name = example['dial_id'], example['turn_id'], example['service_name']
        schema_service = self.schema.services[service_name]

        hist = self.histories[(dial_id, turn_id)]
        hist_pred = self.running_hist_pred[dial_id] if self.eval_mode else self.histories_pred[(dial_id, turn_id)]

        tokenized, other_service_info = input_sequence.construct_input_sequence(
            self.schema, schema_service, example['shuffled_intents'], example['shuffled_slots'],
            example['shuffled_possible_values'], hist, hist_pred, example['uttr'], self.schema_augment_prob)

        usr_uttr_mask = input_sequence.get_usr_uttr_mask(tokenized)
        tokenized = utils.apply_word_dropout(tokenized, usr_uttr_mask, self.word_dropout)

        positions = input_sequence.get_special_token_positions(tokenized)

        tokenized.convert_to_tensors('pt')

        return_example = {
            'dial_id': dial_id,
            'turn_id': turn_id,
            'service_name': service_name,
            'binary': example['binary'],
            'tokenized': tokenized,
            'uttr': example['uttr'],
            **positions,
            'other_service_info': other_service_info,
            'usr_uttr_mask': usr_uttr_mask,
        }

        cross_copy_labels = feature_extraction.get_cross_copy_labels(
            schema_service, example['shuffled_slots'], other_service_info, label['usr_sources'], label['sys_sources'])
        return_label = dict(label, cross_copy=cross_copy_labels)
        return_label.pop('usr_sources')
        return_label.pop('sys_sources')

        return return_example, return_label

    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    utils.create_directories()
    parser = argparse.ArgumentParser(description='Create pytorch datasets.')
    parser.add_argument('--dataset_split')
    parser.add_argument('--eval_mode', action='store_true')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()

    dialogues = sgd_dataset.load_dialogues_from_disk(args.dataset_split)
    schema = sgd_dataset.load_schema_from_disk(args.dataset_split)
    dataset = MultitaskDataset(dialogues, schema, args.eval_mode, args.augment)

    if args.eval_mode:
        file = f'pickles/dataset_{args.dataset_split}_{config.TASK_NAME}_eval.pkl'
    else:
        file = f'pickles/dataset_{args.dataset_split}_{config.TASK_NAME}.pkl'

    with open(file, 'wb') as f:
        pickle.dump(dataset, f)
