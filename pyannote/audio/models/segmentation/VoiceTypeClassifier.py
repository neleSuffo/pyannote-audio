from typing import Optional
import torch
import torch.nn as nn
from pyannote.audio import Model
from pyannote.core import SlidingWindow
from pyannote.audio.core.task import Task, Resolution
from torchaudio.models import wav2vec2_base
from torch.optim.lr_scheduler import _LRScheduler
import math

# Custom scheduler (fallback if CosineScheduler is unavailable)
class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr, learning_rate, min_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.T_max = max_epochs - warmup_epochs
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + (self.learning_rate - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
            return [lr for _ in self.optimizer.param_groups]
        else:
            cosine_epoch = self.last_epoch - self.warmup_epochs
            lr = self.min_lr + 0.5 * (self.learning_rate - self.min_lr) * (
                1 + math.cos(math.pi * cosine_epoch / self.T_max)
            )
            return [lr for _ in self.optimizer.param_groups]

class CustomMultiLabelModel(Model):
    """Sophisticated multilabel segmentation model for ChildLens.SpeakerDiarization.audio
    
    SincNet > Transformer > Feed forward > Classifier
    
    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    task : Task, optional
        Task specification. Defaults to None.
    hidden_size : int, optional
        Hidden size for the transformer encoder. Defaults to 256.
    num_transformer_layers : int, optional
        Number of transformer layers. Defaults to 4.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        hidden_size: int = 256,
        num_transformer_layers: int = 4
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        self.save_hyperparameters("hidden_size", "num_transformer_layers")

        # Pre-trained Wav2Vec 2.0 for feature extraction
        self.feature_extractor = wav2vec2_base()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze weights

        # Transformer encoder with multi-head self-attention
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=768,  # Wav2Vec 2.0 output size
            nhead=8,
            dim_feedforward=hidden_size,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # Linear layer for dimensionality reduction
        self.linear = nn.Linear(768, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def num_frames(self, num_samples: int) -> int:
        # Approximate number of output frames for Wav2Vec 2.0 (20ms stride)
        stride = 320  # 20ms at 16kHz
        return 1 + (num_samples - 512) // stride

    def receptive_field_size(self, num_frames: int = 1) -> int:
        # Approximate receptive field size
        stride = 320  # 20ms at 16kHz
        return num_frames * stride

    def receptive_field(self) -> SlidingWindow:
        # Approximate receptive field for Wav2Vec 2.0
        duration = 0.02  # 20ms per frame
        step = 0.02  # 20ms stride
        return SlidingWindow(start=0.0, duration=duration, step=step)

    def build(self):
        # Add classifier based on task specifications
        num_classes = len(self.specifications.classes)  # 5: kchi, och, mal, fem, ovh
        self.classifier = nn.Linear(self.hparams.hidden_size, num_classes)
        self.activation = self.default_activation()  # Sigmoid for multilabel

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (batch_size, num_channels, num_samples)
        features = self.feature_extractor(waveforms.squeeze(1))  # (batch_size, seq_len, 768)
        features = features.transpose(0, 1)  # (seq_len, batch_size, 768)
        transformer_out = self.transformer(features)  # (seq_len, batch_size, 768)
        output = transformer_out.transpose(0, 1)  # (batch_size, seq_len, 768)
        
        # Apply linear layer and activation
        output = self.relu(self.linear(output))
        output = self.dropout(output)
        
        # Frame-level output for MultiLabelSegmentation
        if self.specifications.resolution == Resolution.FRAME:
            output = self.classifier(output)  # (batch_size, seq_len, num_classes)
            output = self.activation(output)
        
        return output