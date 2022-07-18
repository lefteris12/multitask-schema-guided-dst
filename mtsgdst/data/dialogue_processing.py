import copy

from .. import config
from . import utils
from .classes import *


def parse_system_turn(schema: Schema, turn: Turn) -> Tuple[str, Dict[str, Dict[str, List[str]]]]:
    sys_uttr = ''
    natural_sys_uttr = turn.utterance

    sv_pairs = {}
    for service_name, frame in turn.frames.items():
        service = schema.services[service_name]
        sv_pairs[service_name] = {}
        sys_uttr += ' [SERVICE] ' + service.pr_name
        for action in frame.actions:
            sys_uttr += ' [ACTION] '
            name = action.name
            sys_uttr += name.capitalize().replace('_', ' ')
            if action.slot:
                if name not in ('OFFER_INTENT', 'INFORM_COUNT'):
                    sys_uttr += ' [SLOT] '
                    sys_uttr += action.slot.lower().replace('_', ' ')
                if len(action.values) > 0:
                    sys_uttr += ' [VALUE] '
                    if name == 'OFFER_INTENT':
                        values = service.intents[action.values[0]].pr_name
                    else:
                        values = ' [VALUE] '.join(action.values)
                    sys_uttr += values

            if name in ('INFORM', 'CONFIRM', 'OFFER') or \
                    name == 'REQUEST' and len(action.values) == 1:
                sv_pairs[service_name][action.slot] = action.values

    sys_uttr = sys_uttr.strip()
    sys_uttr = ' '.join(sys_uttr.split())
    sys_uttr = utils.remove_whitespace(sys_uttr)

    if config.USE_NATURAL_SYS_UTTR:
        sys_uttr = natural_sys_uttr

    return sys_uttr, sv_pairs


def parse_service_slot(service, slot, hist, hist_pred, frame):
    service_hist = hist.services[service.name]
    service_hist_pred = hist_pred.services[service.name]

    usr_status, copy_status = utils.USR_STATUS_NONE, utils.COPY_STATUS_NONE
    span = None
    values: List[str] = []
    usr_sources, sys_sources = [], []

    prev_values = service_hist_pred.usr_sv_pairs.get(slot.name, [])

    values_changed = False
    if slot.name in frame.state.sv_pairs:
        values = frame.state.sv_pairs[slot.name]
        # Check for ground truth values
        in_sys_hist_gt = slot.name in service_hist.prev_sys_sv_pairs \
                         and len(service_hist.prev_sys_sv_pairs[slot.name]) == 1 \
                         and values and service_hist.prev_sys_sv_pairs[slot.name][0] in values

        in_sys_uttr_gt = slot.name in service_hist.latest_sys_sv_pairs \
                         and len(service_hist.latest_sys_sv_pairs[slot.name]) == 1 \
                         and values and service_hist.latest_sys_sv_pairs[slot.name][0] in values
        values_changed = (set(prev_values) != set(values)) or (slot.name in frame.slot_spans)

        if values_changed:
            is_slot_informed = False
            for action in frame.actions:
                if action.name == 'INFORM' and action.slot == slot.name:
                    is_slot_informed = True

            if values == ['dontcare']:
                usr_status = utils.USR_STATUS_DONTCARE
            elif slot.is_categorical:
                if is_slot_informed:
                    usr_status = utils.USR_STATUS_ACTIVE
            else:
                if slot.name in frame.state.sv_pairs and slot.name in frame.slot_spans:
                    orig_start, orig_end = frame.slot_spans[slot.name]
                    offset = len(hist.prev_sys_uttr) + len(' [SEP] ')
                    span = (orig_start + offset, orig_end + offset)
                    usr_status = utils.USR_STATUS_ACTIVE

            if usr_status == utils.USR_STATUS_NONE:
                if in_sys_uttr_gt:
                    copy_status = utils.COPY_STATUS_SYS_UTTR
                elif in_sys_hist_gt:
                    copy_status = utils.COPY_STATUS_SYS_HIST
                elif hist.is_multi_domain() and len(set(prev_values) & set(values)) == 0:
                    copy_status = utils.COPY_STATUS_CROSS
                    for o_ser_name, o_ser_hist in hist_pred.services.items():
                        if o_ser_name != service.name:
                            for o_slot_name, o_slot_vals in o_ser_hist.usr_sv_pairs.items():
                                if o_slot_vals and (set(o_slot_vals) & set(values)):
                                    usr_sources.append((o_ser_name, o_slot_name))

                    for o_ser_name, o_ser_hist in hist.services.items():
                        if o_ser_name != service.name:
                            for o_slot_name, o_slot_vals in o_ser_hist.sys_sv_pairs.items():
                                if o_slot_vals and (set(o_slot_vals) & set(values)):
                                    sys_sources.append((o_ser_name, o_slot_name))

        if in_sys_uttr_gt:
            copy_status = utils.COPY_STATUS_SYS_UTTR

    return usr_status, copy_status, span, values, usr_sources, sys_sources, values_changed


def get_new_hist_after_system_turn(hist: DialogueHistory, sys_uttr: str, sv_pairs: Dict[str, Dict[str, List[str]]]) \
        -> None:
    new_hist_gt = hist
    new_hist_gt.prev_sys_uttr = sys_uttr

    for service_name in new_hist_gt.services:
        new_hist_gt.services[service_name].latest_sys_sv_pairs = {}
        new_hist_gt.services[service_name].prev_sys_sv_pairs = copy.deepcopy(
            new_hist_gt.services[service_name].sys_sv_pairs)

    for service_name, service_sv_pairs in sv_pairs.items():
        new_hist_gt.services[service_name].latest_sys_sv_pairs = service_sv_pairs
        new_hist_gt.services[service_name].sys_sv_pairs.update(service_sv_pairs)


def get_new_hist_after_user_turn(hist: DialogueHistory, turn: Turn) -> None:
    new_hist_gt = hist
    current_services = set(turn.frames.keys())
    new_hist_gt.services_so_far.update(turn.frames.keys())
    new_hist_gt.prev_turn_services = current_services


def get_new_service_hist_after_user_turn(
        service_hist_pred: ServiceHistoryPred, intent_name: str, updated_sv_pairs: Dict[str, List[str]]) -> None:
    new_hist_pred = service_hist_pred
    new_hist_pred.intent = intent_name
    new_hist_pred.usr_sv_pairs.update(updated_sv_pairs)
