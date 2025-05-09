import os
import json
import torch
import logging
from pyannote.core import Annotation
from pyannote.audio import Inference
from pyannote.audio.pipelines import MultiLabelSegmentation as MultiLabelSegmentationPipeline
from pyannote.database import registry, FileFinder
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio.models.segmentation import SSeRiouSS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pyannote.core import Annotation # Ensure this import is present if not already

def write_rttm(file_handle, annotation: Annotation, channel_id: int = 1):
    """Write a pyannote.core.Annotation object to an RTTM file stream.

    Parameters
    ----------
    file_handle : file object
        An opened file object for writing.
    annotation : pyannote.core.Annotation
        The annotation to write. Each track's label will be used as the speaker_id.
    channel_id : int, optional
        The channel identifier (field #3 in RTTM). Defaults to 1.
    """
    uri = getattr(annotation, 'uri', 'NA') # Get URI from annotation, default to 'NA' if not set

    for segment, _track_id, speaker_id in annotation.itertracks(yield_label=True):
        start_time = segment.start
        duration = segment.duration

        # RTTM format: type uri channel_id start_time duration <NA> <NA> speaker_id <NA> <NA>
        line = (
            f"SPEAKER {uri} {channel_id} "
            f"{start_time:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker_id} <NA> <NA>\n"
        )
        file_handle.write(line)

def run_inference():
    try:
        # Define paths
        checkpoint_path = "outputs/model_checkpoints/mls_model.ckpt"
        optimized_params_path = "outputs/configs/optimized_pipeline_params.json"
        results_dir = "outputs/inference_results"
        os.makedirs(results_dir, exist_ok=True)

        # Check if model checkpoint and optimized parameters exist
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint not found at {checkpoint_path}")
            return False
        if not os.path.exists(optimized_params_path):
            logger.error(f"Optimized parameters not found at {optimized_params_path}")
            return False

        # Load dataset
        file_finder = FileFinder()
        registry.load_database("/home/nele_pauline_suffo/projects/pyannote-audio/project/data/database.yml")
        protocol = registry.get_protocol(
            'ChildLens.SpeakerDiarization.audio',
            preprocessors={"audio": lambda x: str(file_finder(x))}
        )
        logger.info("Loaded ChildLens dataset")

        # Load model
        mls_model = SSeRiouSS.load_from_checkpoint(checkpoint_path)
        logger.info("Loaded trained model from checkpoint")

        # Initialize pipeline
        pipeline = MultiLabelSegmentationPipeline(
            segmentation=mls_model,
            share_min_duration=True,
            fscore=True
        )
        logger.info("Initialized MultiLabelSegmentationPipeline")

        # Load optimized parameters
        with open(optimized_params_path, 'r') as f:
            optimized_params = json.load(f)
        optimized_pipeline = pipeline.instantiate(optimized_params)
        logger.info("Instantiated pipeline with optimized parameters")

        # Initialize metric
        metric = DiarizationErrorRate()
        logger.info("Initialized DiarizationErrorRate metric")

        # Run inference and evaluation
        for file in protocol.test():
            file_id = file["uri"]
            print(f"Processing file: {file_id}")
            # The pipeline returns an Annotation object
            speech_annotation = optimized_pipeline(file)
            _ = metric(file['annotation'], speech_annotation, uem=file['annotated'])
            output_path = os.path.join(results_dir, f"{file_id}.rttm")
            # Use write_rttm to save the Annotation object
            with open(output_path, "w") as f:
                write_rttm(f, speech_annotation) # Use the defined write_rttm here
            logger.info(f"Saved inference result for {file_id} to {output_path}")

        # Compute and log diarization error rate
        diarization_error_rate = abs(metric) # Corrected variable name
        logger.info(f"Diarization error rate = {diarization_error_rate * 100:.1f}%")
        with open(os.path.join(results_dir, "diarization_error_rate.txt"), "w") as f: # Corrected filename
            f.write(f"Diarization error rate = {diarization_error_rate * 100:.1f}%")
        logger.info("Saved diarization error rate")

        return True

    except Exception as e:
        logger.error(f"Error in run_inference: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting inference")
    success = run_inference()
    if success:
        logger.info("Inference completed successfully")
    else:
        logger.error("Inference failed")