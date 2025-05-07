import torch
from pyannote.database import registry, FileFinder
from pyannote.audio.tasks import MultiLabelSegmentation
from voice_type_classifier import VoiceTypeClassifier
import pytorch_lightning as pl

# Initialize file finder
file_finder = FileFinder()

# Load the ChildLens dataset
registry.load_database("/home/nele_pauline_suffo/projects/pyannote-audio-train/database.yml")
protocol = registry.get_protocol('ChildLens.SpeakerDiarization.audio', preprocessors={"audio": lambda x: str(file_finder(x))})

# Define the MultiLabelSegmentation task
mls_task = MultiLabelSegmentation(
    protocol,
    duration=2.0,
    batch_size=32,
    num_workers=0,  # Avoid multi-process issues on CPU
    classes=['kchi', 'och', 'mal', 'fem', 'ovh']
)

# Create the model with WAVLM_BASE
mls_model = VoiceTypeClassifier(
    task=mls_task,
    hidden_size=256,
    num_transformer_layers=4,
    wav2vec_model="WAVLM_BASE"
)

# Create a PyTorch Lightning trainer and fit the model
trainer = pl.Trainer(
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    max_epochs=1,
    default_root_dir="models/"
)
trainer.fit(mls_model)