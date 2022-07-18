import torch
from torch.utils.data import DataLoader
import argparse
import pickle
import os
import subprocess
import tensorflow as tf

from .schema_guided_dst.baseline import config as baselineconfig, pred_utils_new
from .models import MultitaskModel
from .data import utils, dialogue_processing
from .data.pytorch_dataset import MultitaskDataset, collate_fn
from .data.sgd_dataset import convert_task_name
from . import config


def get_trained_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = MultitaskModel()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(config.DEVICE)
    model.eval()
    return model


def get_model_outputs(logits):
    intent_status = torch.nn.functional.softmax(logits['intent_status'], dim=-1)
    intent_value_probs = torch.nn.functional.softmax(logits['intent_values'], dim=-1)
    usr_status = torch.nn.functional.softmax(logits['usr_status'], dim=-1)
    copy_status = torch.nn.functional.softmax(logits['copy_status'], dim=-1)
    req_status = torch.nn.functional.softmax(logits['req_status'], dim=-1)
    value_probs = torch.nn.functional.softmax(logits['values'], dim=-1)
    cross_probs = torch.nn.functional.softmax(logits['cross'], dim=-1)

    return {
        'intent_status': torch.argmax(intent_status, dim=-1),
        'intent_value_probs': intent_value_probs[:, 1],
        'usr_status': torch.argmax(usr_status, dim=-1),
        'copy_status': torch.argmax(copy_status, dim=-1),
        'req_status': torch.argmax(req_status, dim=-1),
        'value_probs': value_probs[:, 1],
        'start_probs': torch.nn.functional.softmax(logits['start'], dim=-1),
        'end_probs': torch.nn.functional.softmax(logits['end'], dim=-1),
        'cross_probs': cross_probs[:, :, 1]
    }


def get_slot_filling_statuses(model_outputs, idx):
    slot_usr_status = model_outputs['usr_status'][idx].item()
    slot_copy_status = model_outputs['copy_status'][idx].item()
    return slot_usr_status, slot_copy_status


def get_slot_noncat_value(model_outputs, idx, sys_usr_uttr):
    slot_start_probs = model_outputs['start_probs'][idx]
    slot_end_probs = model_outputs['end_probs'][idx]
    slot_start = torch.argmax(slot_start_probs, dim=-1).item()
    slot_end = torch.argmax(slot_end_probs, dim=-1).item()
    if slot_end < slot_start:
        slot_end = slot_start
    tokenized = utils.tokenizer(sys_usr_uttr)

    try:
        c1 = tokenized.token_to_chars(slot_start).start
        c2 = tokenized.token_to_chars(slot_end).end
        output = sys_usr_uttr[c1:c2]
    except:
        output = None

    return output


def get_slot_cat_value(slot, model_outputs, val_idx):
    pos_vals = slot.possible_values
    slot_value_probs = model_outputs['value_probs'][val_idx:val_idx + len(pos_vals)]
    pred_val_idx = torch.argmax(slot_value_probs, dim=-1).item()
    pred_val = pos_vals[pred_val_idx]
    return pred_val


def get_latest_sys_value(service, slot, hist):
    sys_values = hist.services[service.name].sys_sv_pairs.get(slot.name, [])
    if len(sys_values) == 1:
        return sys_values[0]
    return None


def get_slot_cross_copy_value(hist, hist_pred, model_outputs, idx, other_service_info):
    slots_cross_probs = [model_outputs['cross_probs'][idx, i].item() for i, (o_ser_name, o_slot_name) in
                         enumerate(other_service_info) if o_ser_name != 'None' and o_slot_name != 'None']
    if len(slots_cross_probs) > 0:
        cross_copy_idx = max(enumerate(slots_cross_probs), key=lambda x_: x_[1])[0]
        source_serv, source_slot = other_service_info[cross_copy_idx]
        src_service_hist = hist_pred.services[source_serv]
        try:
            value = src_service_hist.usr_sv_pairs[source_slot][0]
        except:
            src_service_hist = hist.services[source_serv]
            value = src_service_hist.sys_sv_pairs[source_slot][0]
        return value
    return None


def intent_prediction(service, hist_pred, model_outputs):
    pos_intents = list(service.intents.keys()) + ['NONE']
    if model_outputs['intent_status']:
        pred_intent_idx = torch.argmax(model_outputs['intent_value_probs'][:len(pos_intents)], dim=-1).item()
        pred_intent = pos_intents[pred_intent_idx]
    else:
        pred_intent = hist_pred.services[service.name].intent

    return pred_intent


def slot_predictions(service, hist, hist_pred, model_outputs, other_service_info, sys_usr_uttr):
    val_idx = 0
    req_slots = []
    slot_status, slot_value = {}, {}

    for idx, slot in enumerate(service.slots.values()):
        req_status = model_outputs['req_status'][idx].item()
        if req_status:
            req_slots.append(slot.name)

        slot_usr_status, slot_copy_status = get_slot_filling_statuses(model_outputs, idx)

        filling_status, value = 'none', None

        if slot.name not in service.usr_slots:
            filling_status, value = 'none', None

        elif slot_usr_status == utils.USR_STATUS_ACTIVE:
            if not slot.is_categorical:
                value = get_slot_noncat_value(model_outputs, idx, sys_usr_uttr)
            else:
                value = get_slot_cat_value(slot, model_outputs, val_idx)

        elif slot_usr_status == utils.USR_STATUS_DONTCARE:
            filling_status = 'dontcare'

        elif slot_copy_status in (utils.COPY_STATUS_SYS_UTTR, utils.COPY_STATUS_SYS_HIST):
            value = get_latest_sys_value(service, slot, hist)

        elif slot_copy_status == utils.COPY_STATUS_CROSS:
            value = get_slot_cross_copy_value(hist, hist_pred, model_outputs, idx, other_service_info)

        if value:
            filling_status = 'active'

        if slot.is_categorical:
            val_idx += len(slot.possible_values)

        slot_status[slot.name] = filling_status
        slot_value[slot.name] = value

    return req_slots, slot_status, slot_value


def process_dialogues(dataset_split, task_name, checkpoint_path):
    model = get_trained_model(checkpoint_path)
    with open(f'pickles/dataset_{dataset_split}_{task_name}_eval.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

    all_predictions = {}
    with torch.no_grad():
        for x, _ in dataloader:
            x = utils.move_to_gpu(x)

            dial_id = x['dial_id'][0]
            turn_id = x['turn_id'][0].item()
            service_name = x['service_name'][0]
            service = val_dataset.schema.services[service_name]

            logits = {k: v[0] for k, v in model(x).items()}
            model_outputs = get_model_outputs(logits)

            hist = val_dataset.histories[(dial_id, turn_id)]
            hist_pred = val_dataset.running_hist_pred[dial_id]

            pred_intent = intent_prediction(service, hist_pred, model_outputs)
            other_service_info = [(v[0][0], v[1][0]) for v in x['other_service_info']]
            req_slots, slot_status, slot_value = slot_predictions(
                service, hist, hist_pred, model_outputs, other_service_info, x['uttr'][0])

            updated_sv_pairs = {k: [v] for k, v in slot_value.items() if v}
            dialogue_processing.get_new_service_hist_after_user_turn(
                hist_pred.services[service_name], pred_intent, updated_sv_pairs)

            all_predictions[(dial_id, str(turn_id).zfill(2), service_name)] = {
                'active_intent': pred_intent,
                'req_slots': req_slots,
                'slot_status': slot_status,
                'slot_value': slot_value
            }

    return all_predictions


def evaluate(predictions, dataset_split, task_name):
    task_name = convert_task_name(task_name)

    dataset_config = baselineconfig.DATASET_CONFIG[task_name]
    input_json_files = [
        os.path.join(config.DATASET_PATH, dataset_split,
                     "dialogues_{:03d}.json".format(fid))
        for fid in dataset_config.file_ranges[dataset_split]
    ]
    schema_json_file = os.path.join(config.DATASET_PATH, dataset_split,
                                    "schema.json")

    # Write predictions to file in DSTC8 format.
    prediction_dir = "temp/pred_res"
    if not tf.io.gfile.exists(prediction_dir):
        tf.io.gfile.makedirs(prediction_dir)
    pred_utils_new.write_predictions_to_file(predictions, input_json_files, schema_json_file, prediction_dir)

    # Evaluate using the scripts provided by the baseline code
    subprocess.call(f"python -m mtsgdst.schema_guided_dst.evaluate --dstc8_data_dir {config.DATASET_PATH} "
                    f"--prediction_dir {prediction_dir} --eval_set {dataset_split} "
                    f"--output_metric_file metric_file.json", shell=True)


def main(dataset_split, checkpoint_path):
    predictions = process_dialogues(dataset_split, config.TASK_NAME, checkpoint_path)
    evaluate(predictions, dataset_split, config.TASK_NAME)


if __name__ == '__main__':
    utils.create_directories()

    parser = argparse.ArgumentParser(description='DST inference.')
    parser.add_argument('--checkpoint_path', default='checkpoints/latest.pt')
    args = parser.parse_args()

    main('test', args.checkpoint_path)
