import torch
import yaml
from pyannote.database import registry, FileFinder
from pyannote.audio.tasks import MultiLabelSegmentation
from pyannote.audio.pipelines import MultiLabelSegmentation as MultiLabelSegmentationPipeline
from pyannote.metrics.detection import DetectionErrorRate
from voice_type_classifier import VoiceTypeClassifier
import pytorch_lightning as pl
from pyannote.pipeline import Optimizer
import pickle

# Initialize file finder
file_finder = FileFinder()

# Load the ChildLens dataset
registry.load_database("/home/nele_pauline_suffo/projects/pyannote-audio/src/database.yml")
protocol = registry.get_protocol('ChildLens.SpeakerDiarization.audio', preprocessors={"audio": lambda x: str(file_finder(x))})

# Verify the split
print(f"Training files: {len(list(protocol.train()))}")
print(f"Validation files: {len(list(protocol.development()))}")
print(f"Test files: {len(list(protocol.test()))}")

# Define the MultiLabelSegmentation task
mls_task = MultiLabelSegmentation(
    protocol,
    duration=2.0,
    batch_size=32,
    num_workers=47,  # Reduced to avoid memory issues
    classes=['kchi', 'och', 'mal', 'fem', 'ovh']
)

# Create the model with WAVLM_BASE
mls_task.prepare_data()
mls_task.setup()
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
    default_root_dir="models/",
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor="loss/val",
            mode="min",
            save_top_k=1,
            dirpath="models/train/ChildLens.SpeakerDiarization.audio",
            filename="best"
        )
    ]
)
trainer.fit(mls_model)

# Save the trained model
trainer.save_checkpoint("models/trained_voice_type_classifier.ckpt")
mls_model.save_hyperparameters("models/hyperparameters.yaml")

# Define the pipeline for multi-label segmentation
pipeline = MultiLabelSegmentationPipeline(segmentation=mls_model, share_min_duration=True, fscore=True)

# Define the initial parameters for the pipeline
initial_params = {
    "thresholds": {
        "kchi": {"onset": 0.6, "offset": 0.4},
        "och": {"onset": 0.6, "offset": 0.4},
        "fem": {"onset": 0.6, "offset": 0.4},
        "mal": {"onset": 0.6, "offset": 0.4},
        "ovh": {"onset": 0.6, "offset": 0.4},
    },
    "min_duration_on": 0.0,
    "min_duration_off": 0.0,
}

# Instantiate the pipeline with the parameters
pipeline.instantiate(initial_params)

# Freeze the pipeline with the specified parameters
pipeline.freeze({'min_duration_on': 0.0, 'min_duration_off': 0.0})

# Optimize the pipeline
optimizer = Optimizer(pipeline)
optimizer.tune(
    list(protocol.development()),
    warm_start=initial_params,
    n_iterations=20,
    show_progress=False
)

optimized_params = optimizer.best_params

# Print and save the optimized parameters
for key, value in optimized_params.items():
    print(f"{key}: {value}")
with open("models/optimized_parameters.txt", "w") as f:
    for key, value in optimized_params.items():
        f.write(f"{key}: {value}\n")

# Save the optimized pipeline
optimized_pipeline = pipeline.instantiate(optimized_params)
with open("models/optimized_pipeline.pkl", "wb") as f:
    pickle.dump(optimized_pipeline, f)

# Evaluate the optimized pipeline on the test set
metric = DetectionErrorRate()
for file in protocol.test():
    speech = optimized_pipeline(file)
    _ = metric(file['annotation'], speech, uem=file['annotated'])

detection_error_rate = abs(metric)
print(f'Detection error rate = {detection_error_rate * 100:.1f}%')