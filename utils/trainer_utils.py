import os
import torch
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

def plot_stats(train_stats, val_stats, num_epochs, title):
    # Unpack statistics
    train_loss, train_acc, train_acc_top_k1, train_acc_top_k2 = zip(*train_stats)
    val_loss, val_acc, val_acc_top_k1, val_acc_top_k2 = zip(*val_stats)

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(32, 6))

    # Plot Loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Accuracy (top 1)
    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy (top1)')
    plt.plot(epochs, val_acc, label='Validation Accuracy (top1)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Top K Accuracy
    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_acc_top_k1, label='Train Top 7 Accuracy')
    plt.plot(epochs, val_acc_top_k1, label='Validation Top 7 Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Top 7 Accuracy')
    plt.title('Training and Validation Top K1 Accuracy')
    plt.legend()

    # Plot Top K2 Accuracy
    plt.subplot(1, 4, 4)
    plt.plot(epochs, train_acc_top_k2, label='Train Top 4 Accuracy')
    plt.plot(epochs, val_acc_top_k2, label='Validation Top 4 Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Top 4 Accuracy')
    plt.title('Training and Validation Top K2 Accuracy')
    plt.legend()

    # Add a big title for the entire figure
    plt.suptitle(title, fontsize=16)
    plt.show()

def print_model_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {num_params} parameters")
    print(f"Total number of parameters: {total_params}")


def calc_metrics(outputs, labels, valid_mask, k1, k2):
    top1_pred_indices = torch.argmax(outputs, dim=-1)
    top1_label_indices = torch.argmax(labels, dim=-1)
    top_k1_label_indices = torch.topk(labels, k=k1, dim=-1)[1]
    top_k2_label_indices = torch.topk(labels, k=k2, dim=-1)[1]
    correct_top_1 = top1_pred_indices == top1_label_indices
    correct_top_k1 = torch.any(torch.eq(top1_pred_indices.unsqueeze(-1), top_k1_label_indices), dim=-1)
    correct_top_k2 = torch.any(torch.eq(top1_pred_indices.unsqueeze(-1), top_k2_label_indices), dim=-1)
    acc_top_1 = torch.sum(correct_top_1 * (valid_mask)) / (valid_mask).sum()
    acc_top_k1 = torch.sum(correct_top_k1 * (valid_mask)) / (valid_mask).sum()
    acc_top_k2 = torch.sum(correct_top_k2 * (valid_mask)) / (valid_mask).sum()
    
    return acc_top_1.item(), acc_top_k1.item(), acc_top_k2.item()

def fetch_hyper_params(params):
    hyper_params = {'batch_size': params['batch_size'],
                    'num_epochs': params['epochs'],
                    'dropout': params['dropout_rate'],
                    'num_transformer_decoder_layers': params['num_transformer_decoder_layers'],
                    'nhead': params['nhead'],
                    'ff_dim': params['transformer_feed_forward_dim'],
                    'decay_factor': params['decay_factor'],
                    'patience': params['decay_LR_patience'],
                    'decay_threshold': params['threshold'],
                    'current_time': datetime.now().strftime("%Y%m%d-%H%M%S")}
    return hyper_params

def fetch_metrics(train_loss, train_acc, train_acc_top_k1, val_loss, val_acc, val_acc_top_k1, train_acc_top_k2, val_acc_top_k2, params):
    metrics = {
            'Loss/train': train_loss,
            'Loss/val': val_loss,
            'Accuracy/train_top_1': train_acc,
            'Accuracy/val_top_1': val_acc,
            f'Accuracy/train_top_{params["k1"]}': train_acc_top_k1,
            f'Accuracy/val_top_{params["k1"]}': val_acc_top_k1,
            f'Accuracy/train_top_{params["k2"]}': train_acc_top_k2,
            f'Accuracy/train_top_{params["k2"]}': val_acc_top_k2,
        }
    return metrics

def get_epoch_message(trainer, train_metrics, val_metrics, epoch):
    message = (f'Epoch {epoch + 1}/{trainer.num_epochs}, '
               f'Train Loss: {train_metrics["loss"]:.4f}, '
               f'Train Acc: {train_metrics["accuracy_top_1"]:.4f}, '
               f'Train Acc Top{trainer.params["k1"]}: {train_metrics["accuracy_top_k1"]:.4f}, '
               f'Train Acc Top{trainer.params["k2"]}: {train_metrics["accuracy_top_k2"]:.4f}, '
               f'Val Loss: {val_metrics["loss"]:.4f}, '
               f'Val Acc: {val_metrics["accuracy_top_1"]:.4f}, '
               f'Val Acc Top{trainer.params["k1"]}: {val_metrics["accuracy_top_k1"]:.4f}, '
               f'Val Acc Top{trainer.params["k2"]}: {val_metrics["accuracy_top_k2"]:.4f}, '
               f'Learning Rate: {trainer.optimizer.param_groups[0]["lr"]:.6f}')
    
    return message

def load_stats(file_path):
    with open(file_path, 'rb') as f:
        train_stats, val_stats, num_epochs = pickle.load(f)
    return train_stats, val_stats, num_epochs


def load_checkpoint(trainer):
    if os.path.isfile(trainer.checkpoint_path):
        checkpoint = torch.load(trainer.checkpoint_path, map_location=trainer.device)
        trainer.network.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.start_epoch = checkpoint['epoch']
        trainer.logger.info(f"Checkpoint loaded: starting from epoch {trainer.start_epoch}")
        trainer.train_stats = checkpoint['train_stats']
        trainer.val_stats = checkpoint['val_stats']

def save_checkpoints(trainer, train_stats, val_stats, epoch):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': trainer.network.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'train_stats': train_stats,
        'val_stats': val_stats
    }
    torch.save(checkpoint, trainer.checkpoint_path)
    trainer.logger.info(f"Checkpoint saved: {trainer.checkpoint_path}")