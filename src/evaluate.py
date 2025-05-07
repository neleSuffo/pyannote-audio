import torch
import yaml
import pickle
from pyannote.database import registry, FileFinder
from pyannote.audio.pipelines import MultiLabelSegmentation as MultiLabelSegmentationPipeline
from pyannote.metrics.detection import DetectionErrorRate
from voice_type_classifier import VoiceTypeClassifier
import pytorch_lightning as pl

# Initialize file finder
file_finder = FileFinder()

# Load the ChildLens dataset
registry.load_database("/home/nele_pauline_suffo/projects/pyannote-audio/src/database.yml")
protocol = registry.get_protocol('ChildLens.SpeakerDiarization.audio', preprocessors={"audio": lambda x: str(file_finder(x))})

# Load the saved model
mls_model = VoiceTypeClassifier.load_from_checkpoint(
    "models/trained_voice_type_classifier.ckpt",
    map_location=torch.device("cpu" if not torch.cuda.is_available() else "cuda")
)

# Load the optimized pipeline
with open("models/optimized_pipeline.pkl", "rb") as f:
    optimized_pipeline = pickle.load(f)

# Evaluate on the test set
metric = DetectionErrorRate()
for file in protocol.test():
    speech = optimized_pipeline(file)
    _ = metric(file['annotation'], speech, uem=file['annotated'])

detection_error_rate = abs(metric)
print(f'Detection error rate = {detection_error_rate * 100:.1f}%')