from exp.data_utils import *
from torch.utils.data.dataloader import default_collate

# This custom collate_fn reduces the input sequence size to the longest input sequence in the batch
def collate_fn(data):
    min_padded = 9999999
    for data_ in data:
        tokenized = data_[0]['tokenized']
        input_ids = list(tokenized['input_ids'])
        padded = input_ids.count(tokenizer.pad_token_id)
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


class AllSlotsDataset:
    def __init__(self, dataset_split, task_name, max_seq_len, use_service_history, use_dialogue_history, eval,
                 use_full_slot_descriptions=False, word_dropout=0, schema_augment_prob=0, use_natural_sys_uttr=0):
        self.dataset_split = dataset_split
        self.task_name = task_name
        self.max_seq_len = max_seq_len
        self.use_service_history = use_service_history
        self.use_dialogue_history = use_dialogue_history
        self.eval = eval
        self.use_full_slot_descriptions = use_full_slot_descriptions
        self.word_dropout = word_dropout if not self.eval else 0
        self.schema_augment_prob = schema_augment_prob if not self.eval else 0

        dialogues = load_dialogues(dataset_split, task_name)
        self.schema = load_schema(dataset_split)
        self.schema_features = create_schema_features(self.schema)

        self.data = create_dialogues_dict(dialogues, self.schema, self.schema_features, use_natural_sys_uttr)

        self.examples = self.create_examples()
        self.num_trunc = 0
        self.num_trunc_noncat = 0

        self.usr_history_sofar = {}
        self.intent_history_sofar = {}

        print(f'Dataset with {len(self.examples)} examples')

    def single_slot_desc(self, service_name, slot_name):
        slot_desc = ' [SERVICE] ' + self.schema_features['service_names'][service_name] + \
                    ' [SLOT] ' + self.schema_features['slot_names'][service_name][slot_name]
        slot_desc = remove_whitespace(slot_desc)
        return slot_desc

    def create_examples(self):
        data = self.data
        schema = self.schema
        examples = []
        for dial_id, d in tqdm(data.items()):
            for turn_id, turn in d.items():
                for service_name, turn_service in turn['services'].items():
                    shuffled_slots = list(schema[service_name]['slots'].keys()).copy()
                    shuffled_intents = list(schema[service_name]['intents'].keys()).copy()
                    shuffled_intents += ['NONE']

                    if not self.eval:
                        random.shuffle(shuffled_slots)
                        random.shuffle(shuffled_intents)

                    example = {
                        'dial_id': dial_id,
                        'turn_id': turn_id,
                        'service_name': service_name,
                        'binary': torch.empty((MAX_SLOTS, NUM_BIN_FEATS)).fill_(0),
                        'shuffled_slots': shuffled_slots,
                        'shuffled_intents': shuffled_intents,
                    }
                    label = {
                        'intent_status': 0,
                        'intent_values': torch.empty(MAX_INTENTS, dtype=torch.int64).fill_(-1),
                        'usr_status': torch.empty(MAX_SLOTS, dtype=torch.int64).fill_(-1),
                        'copy_status': torch.empty(MAX_SLOTS, dtype=torch.int64).fill_(-1),
                        'req_status': torch.empty(MAX_SLOTS, dtype=torch.int64).fill_(-1),
                        'start_idx': torch.empty(MAX_SLOTS, dtype=torch.int64).fill_(-1),
                        'end_idx': torch.empty(MAX_SLOTS, dtype=torch.int64).fill_(-1),
                        'values': torch.empty(MAX_VALUES_PER_SERVICE, dtype=torch.int64).fill_(-1),
                    }

                    num_intents = len(shuffled_intents)
                    label['intent_values'][:num_intents] = 0

                    if turn_service['active_intent'] != turn_service['prev_active_intent']:
                        label['intent_status'] = 1
                        intent_idx = shuffled_intents.index(turn_service['active_intent'])
                        label['intent_values'][intent_idx] = 1

                    values = []
                    shuffled_all_possible_values = []
                    for idx, slot_name in enumerate(shuffled_slots):
                        slot = schema[service_name]['slots'][slot_name]
                        slot_info = turn_service['slots'][slot_name]

                        req_status = slot_name in turn_service['req_slots']

                        binary = torch.Tensor([
                            slot_info['is_new_service'],
                            slot_info['service_switched'],
                            slot_info['in_sys_uttr'],
                            slot_info['in_sys_hist'],
                            slot_info['required_slot'],
                            slot_info['optional_slot'],
                        ])

                        if slot_name in schema[service_name]['usr_slots']:
                            usr_status = slot_info['usr_status']
                            copy_status = slot_info['copy_status']
                            label['usr_status'][idx] = usr_status
                            label['copy_status'][idx] = copy_status
                            if not slot['is_categorical']:
                                span = slot_info['span']
                                slot_desc = self.single_slot_desc(service_name, slot_name)
                                tokenized = tokenizer(turn['sys_usr_uttr'], slot_desc, padding='max_length',
                                                      truncation='only_first',
                                                      max_length=self.max_seq_len)
                                if span:
                                    assert usr_status == USR_STATUS_ACTIVE
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

                                # Debug
                                if not self.eval and span:
                                    tokenized = tokenizer(turn['sys_usr_uttr'])
                                    c1 = tokenized.token_to_chars(start_idx).start
                                    c2 = tokenized.token_to_chars(end_idx).end
                                    assert turn['sys_usr_uttr'][c1:c2] in slot_info['values']
                            else:
                                shuffled_possible_values = slot['possible_values'].copy()
                                if not self.eval:
                                    random.shuffle(shuffled_possible_values)
                                shuffled_all_possible_values += shuffled_possible_values
                                possible_values_len = len(shuffled_possible_values)
                                curr_slot_values = [0] * possible_values_len
                                if usr_status == USR_STATUS_ACTIVE and slot_info['values'][0] != 'dontcare':
                                    assert len(slot_info['values']) == 1
                                    value = slot_info['values'][0]
                                    value_idx = shuffled_possible_values.index(value)
                                    curr_slot_values[value_idx] = 1

                                values += curr_slot_values

                        label['values'] = torch.LongTensor(values + (MAX_VALUES_PER_SERVICE - len(values)) * [-1])
                        label['req_status'][idx] = req_status
                        example['binary'][idx] = binary
                        example['shuffled_possible_values'] = shuffled_all_possible_values
                    examples.append((example, label))
        return examples

    def get_item(self, e):
        example, label = e
        service_name = example['service_name']
        dial_id = example['dial_id']
        turn_id = example['turn_id']
        sys_usr_uttr = self.data[dial_id][turn_id]['sys_usr_uttr']

        schema_desc = '[SERVICE] ' + self.schema_features['service_names'][service_name]

        if self.eval:
            if dial_id not in self.intent_history_sofar:
                prev_intent = 'NONE'
            else:
                prev_intent = self.intent_history_sofar[dial_id].get(service_name, 'NONE')
        else:
            prev_intent = self.data[dial_id][turn_id]['intent_history_sofar'][service_name]

        if self.use_service_history:
            schema_desc += ' Active intent '
            schema_desc += ' : '
            if prev_intent == 'NONE':
                schema_desc += ' [NONE] '
            else:
                schema_desc += self.schema_features['intent_names'][service_name][prev_intent]

        for idx, intent_name in enumerate(example['shuffled_intents']):
            schema_desc += ' [INTENT] '
            if intent_name == 'NONE':
                schema_desc += ' [NONE] '
            else:
                intent_desc = self.schema_features['intent_names'][service_name][intent_name]
                if self.schema_augment_prob > 0:
                    prob = random.uniform(0, 1)
                    if prob < self.schema_augment_prob:
                        intent_desc = augment(intent_desc)
                schema_desc += intent_desc

        value_idx = 0
        for idx, slot_name in enumerate(example['shuffled_slots']):
            schema_desc += ' [SLOT] '

            slot_desc = self.schema_features['slot_names'][service_name][slot_name]
            if slot_name in self.schema[service_name]['usr_slots'] and self.use_full_slot_descriptions:
                slot_desc = self.schema_features['slots'][service_name][slot_name]

            if self.schema_augment_prob > 0:
                prob = random.uniform(0, 1)
                if prob < self.schema_augment_prob:
                    slot_desc = augment(slot_desc)

            schema_desc += slot_desc

            if slot_name not in self.schema[service_name]['usr_slots']:
                continue

            if self.eval:
                usr_hist = self.usr_history_sofar.get(dial_id, {})
            else:
                usr_hist = self.data[dial_id][turn_id]['usr_history_sofar']
            sys_history = self.data[dial_id][turn_id]['curr_sys_history']

            if self.use_service_history:
                schema_desc += ' : '
                if service_name in usr_hist and slot_name in usr_hist[service_name] and usr_hist[service_name][slot_name]:
                    schema_desc += usr_hist[service_name][slot_name][0][0]
                else:
                    schema_desc += ' [NONE] '

            if self.schema[service_name]['slots'][slot_name]['is_categorical']:
                possible_values_len = len(self.schema[service_name]['slots'][slot_name]['possible_values'])
                possible_values = example['shuffled_possible_values'][value_idx:value_idx+possible_values_len]
                value_idx += possible_values_len
                for value in possible_values:
                    value_desc = value
                    if self.schema_augment_prob > 0 and not value.isnumeric():
                        prob = random.uniform(0, 1)
                        if prob < self.schema_augment_prob:
                            value_desc = augment(value)
                    schema_desc += ' [VALUE] ' + value_desc

        cross_copy_labels = torch.empty((MAX_SLOTS, MAX_SLOTS_OTHER_SERVICE), dtype=torch.long).fill_(-1)
        other_service_info = []

        usr_slots_with_values = defaultdict(dict)

        def foo(s):
            if not s['is_categorical']:
                return True
            if s['possible_values'][0].isnumeric():
                return True
            return False

        for other_service_name in sys_history:
            if other_service_name == service_name:
                continue
            for other_slot_name in sys_history[other_service_name]:
                if sys_history[other_service_name][other_slot_name]:
                    if foo(self.schema[other_service_name]['slots'][other_slot_name]):
                        usr_slots_with_values[other_service_name][other_slot_name] = \
                            (sys_history[other_service_name][other_slot_name][0][0], False)

        for other_service_name in usr_hist:
            if other_service_name == service_name:
                continue
            for other_slot_name in usr_hist[other_service_name]:
                if usr_hist[other_service_name][other_slot_name]:
                    if foo(self.schema[other_service_name]['slots'][other_slot_name]):
                        usr_slots_with_values[other_service_name][other_slot_name] = \
                            (usr_hist[other_service_name][other_slot_name][0][0], True)

        shuffled_keys = list(usr_slots_with_values.keys())
        random.shuffle(shuffled_keys)
        other_slot_idx = 0
        for other_service_name in shuffled_keys:
            other_service = usr_slots_with_values[other_service_name]
            schema_desc += ' [SERVICE] '
            schema_desc += self.schema_features['service_names'][other_service_name]
            shuffled_keys2 = list(other_service.keys())
            random.shuffle(shuffled_keys2)

            for other_slot_name in shuffled_keys2:
                other_slot_value, is_usr_slot = other_service[other_slot_name]
                schema_desc += ' [SLOT] '
                if not is_usr_slot:
                    schema_desc += ' system '
                schema_desc += self.schema_features['slot_names'][other_service_name][other_slot_name]
                if self.use_dialogue_history:
                    schema_desc += ' : '
                    schema_desc += str(other_slot_value)

                other_service_info.append((other_service_name, other_slot_name))

                target_slots = self.data[dial_id][turn_id]['services'][service_name]['slots']
                for target_slot_idx, target_slot_name in enumerate(example['shuffled_slots']):
                    if target_slot_name not in self.schema[service_name]['usr_slots']:
                        continue
                    sources = target_slots[target_slot_name]['usr_sources'] + target_slots[target_slot_name]['sys_sources']
                    if (other_service_name, other_slot_name) in sources:
                        cross_copy_labels[target_slot_idx, other_slot_idx] = 1
                    else:
                        cross_copy_labels[target_slot_idx, other_slot_idx] = 0
                other_slot_idx += 1

        schema_desc = remove_whitespace(schema_desc)
        tokenized = tokenizer(sys_usr_uttr, schema_desc, padding='max_length',
                              truncation='only_first',
                              max_length=self.max_seq_len)

        # First [SEP] of the first segment
        start = tokenized['input_ids'].index(tokenizer.sep_token_id) + 1
        end = tokenized['token_type_ids'].index(1) - 2
        usr_uttr_mask = torch.zeros(len(tokenized['input_ids']))
        usr_uttr_mask[start:end+1] = 1

        for idx, start_idx in enumerate(label['start_idx']):
            if start_idx not in (-1, 0) and usr_uttr_mask[start_idx] == 0:
                label['start_idx'][idx] = 0
                self.num_trunc_noncat += 1

        for idx, end_idx in enumerate(label['end_idx']):
            if end_idx not in (-1, 0) and usr_uttr_mask[end_idx] == 0:
                label['end_idx'][idx] = 0
                self.num_trunc_noncat += 1

        if self.word_dropout > 0:
            for idx, input_id in enumerate(tokenized['input_ids']):
                prob = random.uniform(0, 1)
                if usr_uttr_mask[idx] == 1 and prob < self.word_dropout:
                    tokenized['input_ids'][idx] = tokenizer.unk_token_id

        # [SLOT] and [VALUE] tokens positions
        slot_positions = []
        slot_positions_other_service = []
        intent_positions = []
        value_positions = []
        service_tokens_sofar = 0
        for idx, (input_id, type_id) in enumerate(zip(tokenized['input_ids'], tokenized['token_type_ids'])):
            if type_id == 0:
                continue
            if input_id == special_token_name_to_id['[SERVICE]']:
                service_tokens_sofar += 1
            if input_id == special_token_name_to_id['[SLOT]']:
                if service_tokens_sofar == 1:
                    slot_positions.append(idx)
                else:
                    slot_positions_other_service.append(idx)
            elif input_id == special_token_name_to_id['[INTENT]']:
                intent_positions.append(idx)
            elif input_id == special_token_name_to_id['[VALUE]'] and service_tokens_sofar == 1:
                value_positions.append(idx)

        slot_positions += [0] * (MAX_SLOTS - len(slot_positions))
        slot_positions_other_service += [0] * (MAX_SLOTS_OTHER_SERVICE - len(slot_positions_other_service))
        intent_positions += [0] * (MAX_INTENTS - len(intent_positions))
        value_positions += [0] * (MAX_VALUES_PER_SERVICE - len(value_positions))

        slot_positions = torch.LongTensor(slot_positions)
        slot_positions_other_service = torch.LongTensor(slot_positions_other_service)
        intent_positions = torch.LongTensor(intent_positions)
        value_positions = torch.LongTensor(value_positions)
        other_service_info += [('None', 'None')] * (MAX_SLOTS_OTHER_SERVICE - len(other_service_info))

        assert len(slot_positions) == MAX_SLOTS
        assert len(value_positions) == MAX_VALUES_PER_SERVICE

        if tokenized['input_ids'][-1] != 0:
            self.num_trunc += 1
        tokenized.convert_to_tensors('pt')

        return_example = dict(example, tokenized=tokenized,
                              slot_positions=slot_positions,
                              slot_positions_other_service=slot_positions_other_service,
                              intent_positions=intent_positions,
                              value_positions=value_positions,
                              other_service_info=other_service_info,
                              usr_uttr_mask=usr_uttr_mask)
        return_example.pop('shuffled_slots')
        return_example.pop('shuffled_intents')
        return_example.pop('shuffled_possible_values')

        return_label = dict(label, cross_copy=cross_copy_labels)

        return return_example, return_label

    def __getitem__(self, idx):
        example = self.examples[idx]
        return self.get_item(example)

    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Datasets.')
    parser.add_argument('--dataset_split')
    parser.add_argument('--task_name')
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--check_lengths', default=0, type=int)
    parser.add_argument('--use_service_history', default=1, type=int)
    parser.add_argument('--use_dialogue_history', default=1, type=int)
    parser.add_argument('--use_full_slot_descriptions', default=0, type=int)
    parser.add_argument('--use_natural_sys_uttr', default=0, type=int)
    parser.add_argument('--word_dropout', default=0.1, type=float)
    parser.add_argument('--schema_augment_prob', default=0.1, type=float)
    parser.add_argument('--eval', default=0, type=int)
    args = parser.parse_args()

    print(args)
    dataset = AllSlotsDataset(args.dataset_split, args.task_name, args.max_seq_len, args.use_service_history,
                              args.use_dialogue_history, args.eval, args.use_full_slot_descriptions, args.word_dropout,
                              args.schema_augment_prob, args.use_natural_sys_uttr)

    class_name = dataset.__class__.__name__
    if args.eval:
        file = f'pickles/{class_name}_{args.dataset_split}_{args.task_name}_eval.pkl'
    else:
        file = f'pickles/{class_name}_{args.dataset_split}_{args.task_name}.pkl'

    with open(file, 'wb') as f:
        pickle.dump(dataset, f)

    if args.check_lengths:
        for x, y in dataset:
            pass
        print(class_name)
        print(f'Truncated sequences {dataset.num_trunc}')
        print(f'Truncated noncat {dataset.num_trunc_noncat}')
