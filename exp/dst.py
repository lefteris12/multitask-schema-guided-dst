import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import argparse
from sklearn.metrics import classification_report
import copy
import pickle
import os
import subprocess
import tensorflow as tf

from schema_guided_dst.baseline import config, pred_utils_new
from exp import data_utils, dst
from exp.utils import move_to_gpu
from exp.models import AllSlotsModel
from exp.all_slots_dataset import AllSlotsDataset, collate_fn


dataset_dir = "dstc8-schema-guided-dialogue"
output_dir = "temp"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def process_dialogues(dataset_split, task_name, checkpoint_path):
    # Model
    checkpoint = torch.load(checkpoint_path)
    model_args = checkpoint['model_args']
    model = AllSlotsModel(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    # Data
    with open(f'pickles/AllSlotsDataset_{dataset_split}_{task_name}_eval.pkl', 'rb') as f:
        val_dataset = pickle.load(f)

    dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    model_outputs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    intent_outputs = defaultdict(lambda: defaultdict(dict))

    prev_dial_id = None
    # Predictions
    with torch.no_grad():
        usr_status_true, usr_status_pred, copy_status_true, copy_status_pred = [], [], [], []
        cat_correct_cnt, cat_total_cnt = 0, 0
        span_correct_cnt, span_total_cnt = 0, 0
        for x, y in dataloader:
            x = move_to_gpu(x)

            dial_id = x['dial_id'][0]
            turn_id = x['turn_id'][0].item()
            service_name = x['service_name'][0]

            if dial_id != prev_dial_id:
                usr_history = defaultdict(dict)
                val_dataset.intent_history_sofar[dial_id] = defaultdict(dict)

            prev_dial_id = dial_id

            turn = val_dataset.data[dial_id][turn_id]

            logits = {k: v[0] for k, v in model(x).items()}

            intent_status = torch.nn.functional.softmax(logits['intent_status'], dim=-1)
            intent_status = torch.argmax(intent_status, dim=-1)

            intent_value_probs = torch.nn.functional.softmax(logits['intent_values'], dim=-1)
            intent_value_probs = intent_value_probs[:, 1]

            usr_status = torch.nn.functional.softmax(logits['usr_status'], dim=-1)
            usr_status = torch.argmax(usr_status, dim=-1)

            copy_status = torch.nn.functional.softmax(logits['copy_status'], dim=-1)
            copy_status = torch.argmax(copy_status, dim=-1)

            req_status = torch.nn.functional.softmax(logits['req_status'], dim=-1)
            req_status = torch.argmax(req_status, dim=-1)

            value_probs = torch.nn.functional.softmax(logits['values'], dim=-1)
            value_probs = value_probs[:, 1]

            start_probs = torch.nn.functional.softmax(logits['start'], dim=-1)
            end_probs = torch.nn.functional.softmax(logits['end'], dim=-1)

            cross_probs = torch.nn.functional.softmax(logits['cross'], dim=-1)
            cross_probs = cross_probs[:, :, 1]

            if intent_status:
                pos_intents = list(val_dataset.schema[service_name]['intents'].keys()) + ['NONE']
                pred_intent_idx = torch.argmax(intent_value_probs[:len(pos_intents)], dim=-1).item()
                pred_intent = pos_intents[pred_intent_idx]
                val_dataset.intent_history_sofar[dial_id][service_name] = pred_intent
            else:
                pred_intent = val_dataset.intent_history_sofar[dial_id].get(service_name, 'NONE')

            val_idx = 0

            for idx, slot_name in enumerate(val_dataset.schema[service_name]['slots'].keys()):
                slot_req_status = req_status[idx].item()

                status = 'none'
                output = None

                if slot_name in val_dataset.schema[service_name]['usr_slots']:
                    slot_usr_status = usr_status[idx].item()
                    true_usr = y['usr_status'][0][idx]
                    usr_status_pred.append(slot_usr_status)
                    usr_status_true.append(true_usr)

                    slot_copy_status = copy_status[idx].item()
                    true_copy = y['copy_status'][0][idx]
                    copy_status_pred.append(slot_copy_status)
                    copy_status_true.append(true_copy)

                    slot_info = val_dataset.schema[service_name]['slots'][slot_name]
                    true_values = turn['services'][service_name]['slots'][slot_name]['values']
                    if not slot_info['is_categorical']:
                        if slot_usr_status == data_utils.USR_STATUS_ACTIVE:
                            slot_start_probs = start_probs[idx]
                            slot_end_probs = end_probs[idx]
                            slot_start = torch.argmax(slot_start_probs, dim=-1).item()
                            slot_end = torch.argmax(slot_end_probs, dim=-1).item()
                            if slot_end < slot_start:
                                slot_end = slot_start

                            tokenized = data_utils.tokenizer(turn['sys_usr_uttr'])
                            try:
                                c1 = tokenized.token_to_chars(slot_start).start
                                c2 = tokenized.token_to_chars(slot_end).end
                                status = 'active'
                                output = turn['sys_usr_uttr'][c1:c2]
                            except:
                                pass

                            span_total_cnt += 1
                            if (slot_start, slot_end) == (y['start_idx'][0][idx], y['end_idx'][0][idx]):
                                span_correct_cnt += 1
                    else:
                        pos_vals = slot_info['possible_values']
                        slot_value_probs = value_probs[val_idx:val_idx+len(pos_vals)]
                        pred_val_idx = torch.argmax(slot_value_probs, dim=-1).item()
                        val_idx += len(pos_vals)
                        if slot_usr_status == data_utils.USR_STATUS_ACTIVE:
                            pred_val = pos_vals[pred_val_idx]
                            status = 'active'
                            output = pred_val

                            cat_total_cnt += 1
                            if true_values and pred_val in true_values:
                                cat_correct_cnt += 1

                    if slot_usr_status == data_utils.USR_STATUS_ACTIVE:
                        pass
                    elif slot_usr_status == data_utils.USR_STATUS_DONTCARE:
                        status = 'dontcare'
                    elif slot_copy_status in (data_utils.COPY_STATUS_SYS_UTTR, data_utils.COPY_STATUS_SYS_HIST):
                        output = turn['services'][service_name]['slots'][slot_name]['hist_value']
                        if output:
                            status = 'active'
                    elif slot_copy_status == data_utils.COPY_STATUS_CROSS:
                        slots_cross_probs = [cross_probs[idx, i].item() for i, label in
                                             enumerate(y['cross_copy'][0][idx]) if label != -1]
                        if len(slots_cross_probs) > 0:
                            status = 'active'
                            cross_copy_idx = max(enumerate(slots_cross_probs), key=lambda x_: x_[1])[0]
                            (source_serv, ), (source_slot, ) = x['other_service_info'][cross_copy_idx]
                            usr_hist = val_dataset.usr_history_sofar[dial_id]
                            sys_history = val_dataset.data[dial_id][turn_id]['curr_sys_history']
                            try:
                                output = usr_hist[source_serv][source_slot][0][0]
                            except:
                                output = sys_history[source_serv][source_slot][0][0]

                    if status == 'active':
                        usr_history[service_name][slot_name] = [[output]]

                model_outputs[dial_id][turn_id][service_name][slot_name] = (output, status, slot_req_status)
                intent_outputs[dial_id][turn_id][service_name] = pred_intent

            val_dataset.usr_history_sofar[dial_id] = copy.deepcopy(usr_history)

        print('CLASSIFICATION REPORT USER STATUS')
        print(classification_report(usr_status_true, usr_status_pred))
        print('CLASSIFICATION REPORT COPY STATUS')
        print(classification_report(copy_status_true, copy_status_pred))
        print('CAT ACCURACY')
        if cat_total_cnt: print(cat_correct_cnt / cat_total_cnt)
        print('NONCAT SPAN ACCURACY')
        if span_total_cnt: print(span_correct_cnt / span_total_cnt)

    # DST module
    print('DST module...')
    all_predictions = {}
    for dial_id, dial in val_dataset.data.items():
        for turn_id, turn in dial.items():
            for service_name, service in turn['services'].items():
                active = intent_outputs[dial_id][turn_id][service_name]

                slot_status = {}
                slot_value = {}
                req_slots = []
                for slot_name, slot in service['slots'].items():
                    output, status, req_status = model_outputs[dial_id][turn_id][service_name][slot_name]
                    slot_value[slot_name] = output
                    slot_status[slot_name] = status
                    if req_status:
                        req_slots.append(slot_name)

                p = {'active_intent': active, 'req_slots': req_slots, 'slot_status': slot_status,
                     'slot_value': slot_value}

                all_predictions[(dial_id, str(turn_id).zfill(2), service_name)] = p

    return all_predictions


def evaluate(predictions, dataset_split, task_name):
    task_name = data_utils.convert_task_name(task_name)

    dataset_config = config.DATASET_CONFIG[task_name]
    input_json_files = [
        os.path.join(dataset_dir, dataset_split,
                     "dialogues_{:03d}.json".format(fid))
        for fid in dataset_config.file_ranges[dataset_split]
    ]
    schema_json_file = os.path.join(dataset_dir, dataset_split,
                                    "schema.json")

    # Write predictions to file in DSTC8 format.
    prediction_dir = os.path.join(
        output_dir, "pred_res")
    if not tf.io.gfile.exists(prediction_dir):
        tf.io.gfile.makedirs(prediction_dir)
    pred_utils_new.write_predictions_to_file(predictions, input_json_files, schema_json_file, prediction_dir)

    # Evaluate using the scripts provided by the baseline code
    subprocess.call(f"python -m schema_guided_dst.evaluate --dstc8_data_dir {dataset_dir} "
                    f"--prediction_dir {prediction_dir} --eval_set {dataset_split} "
                    f"--output_metric_file metric_file.json", shell=True)


def main(dataset_split, checkpoint_path, task_name='all'):
    predictions = process_dialogues(dataset_split, task_name, checkpoint_path)
    evaluate(predictions, dataset_split, task_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DST inference.')
    parser.add_argument('--task_name', default='all')
    parser.add_argument('--checkpoint_path', default='checkpoints/latest.pt')
    args = parser.parse_args()

    main('test', args.checkpoint_path, args.task_name)
