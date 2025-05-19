import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import torch.optim as optim
from models.network import TractoTransformer
from torch.utils.tensorboard import SummaryWriter
from data_handler import DataHandler
from utils.data.data_utils import build_soft_labels_tensor
from utils.trainer_utils import *
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class TractoTransformerTrainer(object):
    def __init__(self, logger, params, rank, world_size):
        logger.info("Create TractoTransformerTrainer object")

        # General Attributes
        self.params = params
        self.logger = logger
        self.device = params.device

        # Parallelism Attributes
        self.world_size = world_size
        self.rank = rank

        # Create Data Handler
        self.data_handler = DataHandler(params)
        self.train_dwi_data = self.data_handler.train_brains.to(self.device)
        self.val_dwi_data = self.data_handler.val_brains.to(self.device)
        train_sampler = DistributedSampler(self.data_handler.train_dataset, num_replicas=self.world_size, rank=self.rank)
        val_sampler = DistributedSampler(self.data_handler.val_dataset, num_replicas=self.world_size, rank=self.rank)
        self.train_loader = DataLoader(self.data_handler.train_dataset, batch_size=self.params.batch_size, sampler=train_sampler, num_workers=2, pin_memory=False, persistent_workers=True)
        self.val_loader = DataLoader(self.data_handler.val_dataset, batch_size=self.params.batch_size, sampler=val_sampler, num_workers=2, pin_memory=False, persistent_workers=True)
        self.labels_to_soft_labels = build_soft_labels_tensor().to(self.device)

        # Create TractoTransformer Model
        self.network = TractoTransformer(logger, params, self.data_handler.max_sequence_length).to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.network, device_ids=[rank]) 

        # Create optimizer and scheduler
        self.optimizer = Adam(self.network.parameters(), lr=params.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                              factor=params.decay_lr_factor,
                                                              patience=params.decay_lr_patience,
                                                              threshold=params.improvement_threshold,
                                                              threshold_mode='abs',
                                                              min_lr=params.min_lr,
                                                              cooldown=2)

        # Initialize training statistics
        self.train_stats = []
        self.val_stats = []

        # Initialize training parameters
        self.start_epoch = 0
        self.criterion = nn.KLDivLoss(reduction='none')

        if self.params.load_checkpoint:
            load_checkpoint(self)


    def calc_loss(self, outputs, soft_labels, valid_mask):
        """
        Calculate the masked loss using KLDivLoss for padded sequences.

        Parameters:
        - outputs (Tensor): Log probabilities of shape [batch_size, seq_length, 725].
        - labels (Tensor): Integer labels with shape [batch_size, seq_length].
        - valid_mask (Tensor): A boolean tensor of shape [batch_size, seq_length] where True
                                 indicates valid points and False indicates padded points.

        Returns:
        - loss (Tensor): Scalar tensor representing the average loss over all valid points.
        """

        # Calculate the element-wise loss
        elementwise_loss = self.criterion(outputs, soft_labels)

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
            for streamline_voxels_batch, labels, lengths, padding_mask, brain_indices in progress_bar:

                labels = labels.to(self.device)
                streamline_voxels_batch = streamline_voxels_batch.to(self.device)
                padding_mask = padding_mask.to(self.device)

                # Forward pass
                outputs = self.network(self.train_dwi_data, streamline_voxels_batch, padding_mask, brain_indices)
                soft_labels = self.labels_to_soft_labels[labels]
                loss = self.calc_loss(outputs, soft_labels, ~padding_mask)
                acc_top_1, acc_top_k1, acc_top_k2 = calc_metrics(outputs, soft_labels, ~padding_mask, self.params.k1, self.params.k2)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log loss and top-k accuracies
                total_loss += loss.item()
                total_acc_top_1 += acc_top_1
                total_acc_top_k1 += acc_top_k1
                total_acc_top_k2 += acc_top_k2

                progress_bar.set_postfix({'loss': loss.item(),
                                          'acc': acc_top_1,
                                          f'top{self.params.k1}': acc_top_k1,
                                          f'top{self.params.k2}': acc_top_k2})


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
        self.logger.info("TractoTransformerTrainer: Validation phase")
        self.network.eval()
        total_loss, total_acc_top_1, total_acc_top_k1, total_acc_top_k2 = 0, 0, 0, 0
        with torch.no_grad():
            for streamline_voxels_batch, labels, lengths, padding_mask, brain_indices in data_loader:

                labels = labels.to(self.device)
                streamline_voxels_batch = streamline_voxels_batch.to(self.device)
                padding_mask = padding_mask.to(self.device)

                # Forward pass
                outputs = self.network(self.val_dwi_data, streamline_voxels_batch, padding_mask, brain_indices)
                soft_labels = self.labels_to_soft_labels[labels]
                loss = self.calc_loss(outputs, soft_labels, ~padding_mask)
                acc_top_1, acc_top_k1, acc_top_k2 = calc_metrics(outputs, soft_labels, ~padding_mask, self.params.k1, self.params.k2)

                # Log loss and top-k accuracies
                total_loss += loss.item()
                total_acc_top_1 += acc_top_1
                total_acc_top_k1 += acc_top_k1
                total_acc_top_k2 += acc_top_k2


        val_loss = total_loss / len(data_loader)
        val_acc_top_1 = total_acc_top_1 / len(data_loader)
        val_acc_top_k1 = total_acc_top_k1 / len(data_loader)
        val_acc_top_k2 = total_acc_top_k2 / len(data_loader)

        if self.params.decay_lr:
            self.scheduler.step(val_loss)

        return {'loss': val_loss,
                'accuracy_top_1': val_acc_top_1,
                'accuracy_top_k1': val_acc_top_k1,
                'accuracy_top_k2': val_acc_top_k2 
                }


    def train(self):
        train_stats, val_stats = self.train_stats, self.val_stats
        for epoch in range(self.start_epoch, self.params.epochs):
            self.train_sampler.set_epoch(epoch)
            self.logger.info("TractoTransformerTrainer: Training Epoch")
            train_metrics = self.train_epoch(self.train_loader)
            val_metrics = self.validate(self.val_loader)

            # Print epoch message
            if self.rank == 0:
                self.logger.info(get_epoch_message(self, train_metrics, val_metrics, epoch))

                # Save statistics
                train_stats.append((train_metrics['loss'], train_metrics['accuracy_top_1'], train_metrics['accuracy_top_k1'], train_metrics['accuracy_top_k2']))
                val_stats.append((val_metrics['loss'], val_metrics['accuracy_top_1'], val_metrics['accuracy_top_k1'], val_metrics['accuracy_top_k2']))
                
            # Save checkpoints
                if self.params.save_checkpoints:
                    save_checkpoints(self, train_stats, val_stats, epoch)

        if self.rank == 0 and self.params.save_checkpoints:
            save_checkpoints(self, train_stats, val_stats, epoch+1)
            if self.params.save_model:
                save_model(self)

        return train_stats, val_stats
