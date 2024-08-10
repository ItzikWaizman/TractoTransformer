import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from torch.optim import Adam
import torch.optim as optim
from models.network import TractoTransformer
from torch.utils.tensorboard import SummaryWriter
from data_handling import *
from utils.trainer_utils import *
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.nn.parallel import DistributedDataParallel as DDP


class TractoGNNTrainer(object):
    def __init__(self, logger, params, rank, world_size):
        logger.info("Create TractoGNNTrainer object")
        self.logger = logger
        self.device = params['device']
        self.world_size = world_size
        self.rank = rank
        self.network = TractoTransformer(logger=logger, params=params).to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.network, device_ids=[rank])
        self.train_data_handler = SubjectDataHandler(logger=logger, params=params, mode=TRAIN, device=self.device)
        self.val_data_handler = SubjectDataHandler(logger=logger, params=params, mode=VALIDATION, device=self.device)
        self.train_dwi_data = self.train_data_handler.dwi.to(self.device)
        self.val_dwi_data = self.val_data_handler.dwi.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=params['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                              factor=params['decay_factor'],
                                                              patience=params['decay_LR_patience'],
                                                              threshold=params['threshold'],
                                                              threshold_mode='abs',
                                                              min_lr=params['min_lr'],
                                                              cooldown=2)
        self.train_stats = []
        self.val_stats = []
        self.num_epochs = params['epochs']
        self.start_epoch = 0
        self.criterion = nn.KLDivLoss(reduction='none')
        self.train_causality_mask = self.train_data_handler.causality_mask.to(self.device)
        self.val_causality_mask = self.val_data_handler.causality_mask.to(self.device)
        self.trained_model_path = params['trained_model_path']
        self.checkpoint_path = params['checkpoint_path']
        self.params = params
        train_dataset = self.train_data_handler.dataset
        val_dataset = self.val_data_handler.dataset
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank) 
        self.train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], sampler=train_sampler, num_workers=2, pin_memory=False, persistent_workers=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'], sampler=val_sampler, num_workers=2, pin_memory=False, persistent_workers=True)
        if params['load_checkpoint']:
            load_checkpoint(self)

    def calc_loss(self, outputs, labels, valid_mask):
        """
        Calculate the masked loss using KLDivLoss for padded sequences.

        Parameters:
        - outputs (Tensor): Log probabilities of shape [batch_size, seq_length, 725].
        - labels (Tensor): True probabilities with shape [batch_size, seq_length, 725].
        - valid_mask (Tensor): A boolean tensor of shape [batch_size, seq_length] where True
                                 indicates valid points and False indicates padded points.

        Returns:
        - loss (Tensor): Scalar tensor representing the average loss over all valid points.
        """

        # Calculate the element-wise loss
        elementwise_loss = self.criterion(outputs, labels)

        # Apply the padding mask to ignore loss for padded values
        # We need to unsqueeze the padding_mask to make it broadcastable to elementwise_loss shape
        masked_loss = elementwise_loss * valid_mask.unsqueeze(-1)
        
        # Calculate the average loss per valid sequence element
        loss = masked_loss.sum() / valid_mask.sum()

        return loss

    def train_epoch(self, data_loader):
        self.network.train()
        total_loss, total_acc_top_1, total_acc_top_k1, total_acc_top_k2 = 0, 0, 0, 0
        with tqdm(data_loader, desc='Training', unit='batch') as progress_bar:
            for streamline_voxels_batch, labels, lengths, padding_mask in progress_bar:
                labels = labels
                streamline_voxels_batch = streamline_voxels_batch
                padding_mask = padding_mask

                # Forward pass
                outputs = self.network(self.train_dwi_data, streamline_voxels_batch, padding_mask, self.train_causality_mask)
                loss = self.calc_loss(outputs, labels, ~padding_mask)
                acc_top_1, acc_top_k1, acc_top_k2 = calc_metrics(outputs, labels, ~padding_mask, self.params['k1'], self.params['k2'])

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_acc_top_1 += acc_top_1
                total_acc_top_k1 += acc_top_k1
                total_acc_top_k2 += acc_top_k2

                progress_bar.set_postfix({'loss': loss.item(),
                                          'acc': acc_top_1,
                                          f'top{self.params["k1"]}': acc_top_k1,
                                          f'top{self.params["k2"]}': acc_top_k2})


        train_loss = total_loss / len(data_loader)
        train_acc_top_1 = total_acc_top_1 / len(data_loader)
        train_acc_top_k1 = total_acc_top_k1 / len(data_loader)
        train_acc_top_k2 = total_acc_top_k2 / len(data_loader)

        return {'loss': train_loss,
                'accuracy_top_1': train_acc_top_1,
                'accuracy_top_k1': train_acc_top_k1,
                'accuracy_top_k2': train_acc_top_k2 
                }

    def validate(self, data_loader):
        self.logger.info("TractoGNNTrainer: Validation phase")
        self.network.eval()
        total_loss, total_acc_top_1, total_acc_top_k1, total_acc_top_k2 = 0, 0, 0, 0
        with torch.no_grad():
            for streamline_voxels_batch, labels, lengths, padding_mask in data_loader:
                labels = labels
                streamline_voxels_batch = streamline_voxels_batch
                padding_mask = padding_mask

                # Forward pass
                outputs = self.network(self.val_dwi_data, streamline_voxels_batch, padding_mask, self.val_causality_mask)
                loss = self.calc_loss(outputs, labels, ~padding_mask)
                acc_top_1, acc_top_k1, acc_top_k2 = calc_metrics(outputs, labels, ~padding_mask, self.params['k1'], self.params['k2'])

                total_loss += loss.item()
                total_acc_top_1 += acc_top_1
                total_acc_top_k1 += acc_top_k1
                total_acc_top_k2 += acc_top_k2

        val_loss = total_loss / len(data_loader)
        val_acc_top_1 = total_acc_top_1 / len(data_loader)
        val_acc_top_k1 = total_acc_top_k1 / len(data_loader)
        val_acc_top_k2 = total_acc_top_k2 / len(data_loader)

        if self.params['decay_LR']:
            self.scheduler.step(val_loss)

        return {'loss': val_loss,
                'accuracy_top_1': val_acc_top_1,
                'accuracy_top_k1': val_acc_top_k1,
                'accuracy_top_k2': val_acc_top_k2 
                }

    def train(self):
        if self.rank == 0:
            log_dir = "logs"
            writer = SummaryWriter(log_dir=log_dir)
            writer.add_hparams(fetch_hyper_params(self.params), {})
            writer.add_text('FODFs prediction', 'This is an experiment ONE fiber bundle', 0)

        train_stats, val_stats = self.train_stats, self.val_stats
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info("TractoGNNTrainer: Training Epoch")
            train_metrics = self.train_epoch(self.train_loader)
            val_metrics = self.validate(self.val_loader)

            # Print epoch message
            if self.rank == 0:
                self.logger.info(get_epoch_message(self, train_metrics, val_metrics, epoch))

                # Log metrics
                for metric_name, metric_value in train_metrics.items():
                    writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
            
                for metric_name, metric_value in val_metrics.items():
                    writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)

                # Save statistics
                train_stats.append((train_metrics['loss'], train_metrics['accuracy_top_1'], train_metrics['accuracy_top_k1'], train_metrics['accuracy_top_k2']))
                val_stats.append((val_metrics['loss'], val_metrics['accuracy_top_1'], val_metrics['accuracy_top_k1'], val_metrics['accuracy_top_k2']))
                
            # Save checkpoints
                if self.params['save_checkpoints']:
                    save_checkpoints(self, train_stats, val_stats, epoch)

        if self.rank == 0:
            save_checkpoints(self, train_stats, val_stats, epoch+1)
            writer.flush()
            writer.close()
        return train_stats, val_stats
