# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prediction and evaluation-related utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

import tensorflow.compat.v1 as tf

from schema_guided_dst import schema
from schema_guided_dst.baseline import data_utils


REQ_SLOT_THRESHOLD = 0.5


def get_predicted_dialog(dialog, all_predictions, schemas):
  """Update labels in a dialogue based on model predictions.

  Args:
    dialog: A json object containing dialogue whose labels are to be updated.
    all_predictions: A dict mapping prediction name to the predicted value. See
      SchemaGuidedDST class for the contents of this dict.
    schemas: A Schema object wrapping all the schemas for the dataset.

  Returns:
    A json object containing the dialogue with labels predicted by the model.
  """
  # Overwrite the labels in the turn with the predictions from the model. For
  # test set, these labels are missing from the data and hence they are added.
  dialog_id = dialog["dialogue_id"]
  # The slot values tracked for each service.
  all_slot_values = collections.defaultdict(dict)
  for turn_idx, turn in enumerate(dialog["turns"]):
    if turn["speaker"] == "USER":
      user_utterance = turn["utterance"]
      system_utterance = (
          dialog["turns"][turn_idx - 1]["utterance"] if turn_idx else "")
      turn_id = "{:02d}".format(turn_idx)
      for frame in turn["frames"]:
        predictions = all_predictions[(dialog_id, turn_id, frame["service"])]
        slot_values = all_slot_values[frame["service"]]
        service_schema = schemas.get_service_schema(frame["service"])
        # Remove the slot spans and state if present.
        frame.pop("slots", None)
        frame.pop("state", None)

        # The baseline model doesn't predict slot spans. Only state predictions
        # are added.
        state = {}

        # Add prediction for active intent. Offset is subtracted to account for
        # NONE intent.
        state["active_intent"] = predictions["active_intent"]

        # Add prediction for requested slots.
        state["requested_slots"] = predictions["req_slots"]

        # Add prediction for user goal (slot values).
        # Categorical slots.
        for slot_idx, slot in enumerate(service_schema.categorical_slots):
          slot_status = predictions["slot_status"][slot]
          if slot_status == 'dontcare':
            slot_values[slot] = data_utils.STR_DONTCARE
          elif slot_status == 'active':
            slot_values[slot] = predictions["slot_value"][slot]
        # Non-categorical slots.
        for slot_idx, slot in enumerate(service_schema.non_categorical_slots):
          slot_status = predictions["slot_status"][slot]
          if slot_status == 'dontcare':
            slot_values[slot] = data_utils.STR_DONTCARE
          elif slot_status == 'active':
            slot_values[slot] = predictions["slot_value"][slot]
        # Create a new dict to avoid overwriting the state in previous turns
        # because of use of same objects.
        state["slot_values"] = {s: [v] for s, v in slot_values.items()}
        frame["state"] = state
  return dialog


def write_predictions_to_file(predictions, input_json_files, schema_json_file,
                              output_dir):
  """Write the predicted dialogues as json files.

  Args:
    predictions: An iterator containing model predictions. This is the output of
      the predict method in the estimator.
    input_json_files: A list of json paths containing the dialogues to run
      inference on.
    schema_json_file: Path for the json file containing the schemas.
    output_dir: The directory where output json files will be created.
  """
  tf.compat.v1.logging.info("Writing predictions to %s.", output_dir)
  schemas = schema.Schema(schema_json_file)
  # Index all predictions.
  all_predictions = predictions

  # Read each input file and write its predictions.
  for input_file_path in input_json_files:
    with tf.io.gfile.GFile(input_file_path) as f:
      dialogs = json.load(f)
      pred_dialogs = []
      for d in dialogs:
        pred_dialogs.append(get_predicted_dialog(d, all_predictions, schemas))
    input_file_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_dir, input_file_name)
    with tf.io.gfile.GFile(output_file_path, "w") as f:
      json.dump(
          pred_dialogs, f, indent=2, separators=(",", ": "), sort_keys=True)
