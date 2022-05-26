# Dependencies

```
pip install -r requirements.txt
```

Using a virtual environment is recommended.

# Create datasets

Download the SGD dataset:
```
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
```

Create and save the datasets to pickles.
The `eval` flag controls whether the dataset uses the ground-truth previous dialogue states:

```
python -m exp.all_slots_dataset --dataset_split=train --task_name=all --schema_augment_prob=0.1 --word_dropout=0.1
python -m exp.all_slots_dataset --dataset_split=dev --task_name=all --schema_augment_prob=0 --word_dropout=0
python -m exp.all_slots_dataset --dataset_split=dev --task_name=all --schema_augment_prob=0 --word_dropout=0 --eval=1
python -m exp.all_slots_dataset --dataset_split=test --task_name=all --schema_augment_prob=0 --word_dropout=0 --eval=1
```

# Training

Train the model for a total of 5 epochs and evaluate every 4k steps on the dev set:

```
python -m exp.train --num_epochs=5 --num_total_epochs=5 --batch_size=16 --dropout=0.3 --num_steps_logging=4000
```

After every evaluation on the dev set, `checkpoints/latest.pt` is the latest checkpoint and `checkpoints/best.pt` is the best checkpoint so far.
   
It is possible to continue training from checkpoints.
For example:

```
python -m exp.train --num_epochs=2 --num_total_epochs=5 --batch_size=16 --dropout=0.3 --num_steps_logging=4000
python -m exp.train --num_epochs=5 --num_total_epochs=5 --batch_size=16 --dropout=0.3 --num_steps_logging=4000 --load_checkpoint_path='checkpoints/latest.pt'
```

# Evaluation on the test set

Evaluate on the test set using the [scripts](https://github.com/google-research/google-research/tree/master/schema_guided_dst) provided by the SGD-baseline:
```
python -m exp.dst --checkpoint_path='checkpoints/best.pt'
```
