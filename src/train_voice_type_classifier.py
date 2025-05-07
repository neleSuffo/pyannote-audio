import torch
from pyannote.database import registry, FileFinder
from pyannote.audio.tasks import MultiLabelSegmentation
from pyannote.audio.pipelines import MultiLabelSegmentation as MultiLabelSegmentationPipeline
from voice_type_classifier import VoiceTypeClassifier
import pytorch_lightning as pl
from pyannote.pipeline import Optimizer

# Initialize file finder
file_finder = FileFinder()

# Load the ChildLens dataset
registry.load_database("/home/nele_pauline_suffo/projects/pyannote-audio/src/database.yml")
protocol = registry.get_protocol('ChildLens.SpeakerDiarization.audio', preprocessors={"audio": lambda x: str(file_finder(x))})

# Define the MultiLabelSegmentation task
mls_task = MultiLabelSegmentation(
    protocol,
    cache="/home/nele_pauline_suffo/ProcessedData/vtc_childlens/cache/task_cache.npz",
    duration=2.0,
    batch_size=32,
    num_workers=47,
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

# define the pipeline for multi-label segmentation
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

optimizer = Optimizer(pipeline)
optimizer.tune(list(protocol.development()),
               warm_start=initial_params,
               n_iterations=20,
               show_progress=False)

optimized_params = optimizer.best_params

# print and save the optimized parameters
for key, value in optimized_params.items():
    print(f"{key}: {value}")
with open("optimized_parameters.txt", "w") as f:
    for key, value in optimized_params.items():
        f.write(f"{key}: {value}\n")
        
optimized_pipeline = pipeline.instantiate(optimized_params)

# Evaluate the optimized pipeline
metric = DetectionErrorRate()

for file in protocol.test():
    speech = optimized_pipeline(file)
    _ = metric(file['annotation'], speech, uem=file['annotated'])

detection_error_rate = abs(metric)
print(f'Detection error rate = {detection_error_rate * 100:.1f}%')