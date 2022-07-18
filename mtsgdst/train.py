import pickle
import argparse
import json

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformers import AdamW, get_scheduler
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import MultitaskModel
from .data.pytorch_dataset import MultitaskDataset, collate_fn
from .data import utils
from . import dst, config


def get_loss(y_pred, y_true):
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)

    loss_intent_status = cross_entropy(y_pred['intent_status'], y_true['intent_status']).nan_to_num()
    loss_intent_values = cross_entropy(
        y_pred['intent_values'].reshape(-1, 2), y_true['intent_values'].reshape(-1)).nan_to_num()

    loss_usr_status = cross_entropy(
        y_pred['usr_status'].reshape(-1, 3), y_true['usr_status'].reshape(-1)).nan_to_num()
    loss_copy_status = cross_entropy(
        y_pred['copy_status'].reshape(-1, 4), y_true['copy_status'].reshape(-1)).nan_to_num()
    loss_req_status = cross_entropy(
        y_pred['req_status'].reshape(-1, 2), y_true['req_status'].reshape(-1)).nan_to_num()

    loss_values = cross_entropy(y_pred['values'].reshape(-1, 2), y_true['values'].reshape(-1)).nan_to_num()
    loss_cat = loss_values

    loss_start = cross_entropy(
        y_pred['start'].reshape(-1, y_pred['start'].shape[-1]), y_true['start_idx'].reshape(-1)).nan_to_num()
    loss_end = cross_entropy(
        y_pred['end'].reshape(-1, y_pred['end'].shape[-1]), y_true['end_idx'].reshape(-1)).nan_to_num()
    loss_noncat = (loss_start + loss_end) / 2

    loss_cross = cross_entropy(y_pred['cross'].reshape(-1, 2), y_true['cross_copy'].reshape(-1)).nan_to_num()

    loss_dst = (3 * loss_usr_status + 3 * loss_copy_status + loss_cat + loss_noncat + loss_cross) / 9
    loss_intent = (2 * loss_intent_status + loss_intent_values) / 3
    loss = (7 * loss_dst + 1 * loss_intent + 1 * loss_req_status) / 9
    return loss


def train(model):
    def save_checkpoint(path):
        print(f'Saving checkpoint for epoch {epoch}')
        torch.save({
            'epoch': epoch,
            'val_losses': val_losses,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'max_val_acc': max_val_acc,
        }, path)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

    if args.load_checkpoint_path:
        checkpoint = torch.load(args.load_checkpoint_path)
        starting_epoch = checkpoint['epoch'] + 1
        val_losses = checkpoint['val_losses']
        max_val_acc = checkpoint['max_val_acc']
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'Loading checkpoint and continuing training from epoch {starting_epoch}')
    else:
        starting_epoch = 1
        val_losses = []
        max_val_acc = -1

    model.to(config.DEVICE)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    num_training_steps = args.num_total_epochs * len(train_dataloader)
    num_warmup_steps = 0.1 * num_training_steps
    lr_scheduler = get_scheduler(config.LR_SCHEDULER, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps)
    if args.load_checkpoint_path:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    with open('temp/model_architecture.txt', 'w') as f:
        f.write(str(model))

    print(f'=== TRAINING ===')
    print('Script called with args:')
    print(args)
    print(f'=== TRAINING ===')

    for epoch in range(starting_epoch, args.num_epochs + 1):
        model.train()
        print(f'Epoch: {epoch}')
        running_train_loss = 0.0
        for step, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            model.train()
            x = utils.move_to_gpu(x)
            y = utils.move_to_gpu(y)
            optimizer.zero_grad()
            outputs = model(x)
            loss = get_loss(outputs, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            running_train_loss += loss.item()

            # Calculate validation loss
            if (args.num_steps_logging and (step + 1) % args.num_steps_logging == 0) \
                    or step == len(train_dataloader) - 1:
                model.eval()
                running_val_loss = 0.0
                for x_val, y_val in val_dataloader:
                    x_val = utils.move_to_gpu(x_val)
                    y_val = utils.move_to_gpu(y_val)
                    with torch.no_grad():
                        outputs = model(x_val)
                        loss = get_loss(outputs, y_val)
                        running_val_loss += loss.item()
                val_loss = running_val_loss / len(val_dataloader)
                val_losses.append(val_loss)
                print(f'\nValidation loss: {val_loss}')

                save_checkpoint('checkpoints/latest.pt')
                # Evaluate on the development set
                dst.main('dev', 'checkpoints/latest.pt')

                with open('metric_file.json', 'r') as f:
                    json_metric_file = json.load(f)
                    val_acc = json_metric_file['#ALL_SERVICES']['joint_goal_accuracy']
                    val_unseen_acc = json_metric_file['#UNSEEN_SERVICES']['joint_goal_accuracy']
                    print(f'Validation accuracy: {val_acc}')
                    print(f'Validation unseen accuracy: {val_unseen_acc}')

                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    save_checkpoint('checkpoints/best.pt')

                save_checkpoint('checkpoints/latest.pt')

        train_loss = running_train_loss / len(train_dataloader)
        print(f'\nTrain loss: {train_loss}')

    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='validation')
    plt.legend()
    plt.savefig('figs/losses.png')


if __name__ == '__main__':
    utils.create_directories()
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--load_checkpoint_path')
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--num_total_epochs', default=3, type=int)
    parser.add_argument('--num_steps_logging', type=int)
    args = parser.parse_args()

    model = MultitaskModel()
    print(model)

    with open(f'pickles/dataset_train_{config.TASK_NAME}.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(f'pickles/dataset_dev_{config.TASK_NAME}.pkl', 'rb') as f:
        val_dataset = pickle.load(f)

    train(model)
