from typing import Optional
import torch
import torch.nn as nn
import torchaudio
from pyannote.audio import Model
from pyannote.core import SlidingWindow
from pyannote.audio.core.task import Task, Resolution
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

class VoiceTypeClassifier(Model):
    """Sophisticated multilabel segmentation model for ChildLens.SpeakerDiarization.audio
    
    wav2vec > Transformer > Feed forward > Classifier
    
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
    wav2vec_model : str, optional
        Pre-trained wav2vec model from torchaudio.pipelines. Defaults to "WAVLM_BASE".
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        hidden_size: int = 256,
        num_transformer_layers: int = 4,
        wav2vec_model: str = "WAVLM_BASE"
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        self.save_hyperparameters("hidden_size", "num_transformer_layers", "wav2vec_model")

        # Load pre-trained WavLM_BASE (or other wav2vec model)
        if hasattr(torchaudio.pipelines, self.hparams.wav2vec_model):
            bundle = getattr(torchaudio.pipelines, self.hparams.wav2vec_model)
            if sample_rate != bundle._sample_rate:
                raise ValueError(
                    f"Expected {bundle._sample_rate}Hz, found {sample_rate}Hz."
                )
            self.wav2vec = bundle.get_model()
            wav2vec_dim = bundle._params["encoder_embed_dim"]
        else:
            raise ValueError(f"Unsupported wav2vec model: {self.hparams.wav2vec_model}")

        # Freeze wav2vec weights
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        # Transformer encoder with multi-head self-attention
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=wav2vec_dim,  # Match wav2vec output size
            nhead=8,
            dim_feedforward=hidden_size,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # Linear layer for dimensionality reduction
        self.linear = nn.Linear(wav2vec_dim, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        # Initialize task-dependent layers
        if task is not None:
            self.build()

    # @property
    # def example_input_array(self) -> torch.Tensor:
    #     """Define a dummy input for caching example_output"""
    #     # 2-second audio chunk at 16kHz, single channel
    #     return torch.randn(1, self.hparams.num_channels, self.hparams.sample_rate * 2)

    # @property
    # def example_output(self) -> torch.Tensor:
    #     """Explicitly define example_output to ensure compatibility"""
    #     if not hasattr(self, 'classifier'):
    #         # Fallback output if classifier is not yet defined
    #         num_frames = self.num_frames(self.hparams.sample_rate * 2)
    #         return torch.zeros(1, num_frames, 5)  # Assuming 5 classes
    #     return self(self.example_input_array)

    def num_frames(self, num_samples: int) -> int:
        # Approximate number of output frames for WavLM_BASE (20ms stride)
        stride = 320  # 20ms at 16kHz
        return 1 + (num_samples - 512) // stride

    def receptive_field_size(self, num_frames: int = 1) -> int:
        # Approximate receptive field size
        stride = 320  # 20ms at 16kHz
        return num_frames * stride

    def receptive_field(self) -> SlidingWindow:
        # Approximate receptive field for WavLM_BASE
        duration = 0.02  # 20ms per frame
        step = 0.02  # 20ms stride
        return SlidingWindow(start=0.0, duration=duration, step=step)

    def build(self):
        # Add classifier based on task specifications
        num_classes = len(self.specifications.classes)  # 5: kchi, och, mal, fem, ovh
        self.classifier = nn.Linear(self.hparams.hidden_size, num_classes)
        self.activation = self.default_activation()  # Sigmoid for multilabel

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        # Extract wav2vec features with frozen weights
        with torch.no_grad():
            outputs, _ = self.wav2vec.extract_features(waveforms.squeeze(1))
            features = outputs[-1]  # Use last layer (batch, seq_len, wav2vec_dim)

        # Transformer processing
        features = features.transpose(0, 1)  # (seq_len, batch, wav2vec_dim)
        transformer_out = self.transformer(features)  # (seq_len, batch, wav2vec_dim)
        output = transformer_out.transpose(0, 1)  # (batch, seq_len, wav2vec_dim)
        
        # Apply linear layer and activation
        output = self.relu(self.linear(output))
        output = self.dropout(output)
        
        # Frame-level output for MultiLabelSegmentation
        if self.specifications.resolution == Resolution.FRAME:
            output = self.classifier(output)  # (batch, seq_len, num_classes)
            output = self.activation(output)
        
        return output