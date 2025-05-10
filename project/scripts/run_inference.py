import os
import json
import torch
import logging
import argparse
from pyannote.core import Annotation
from pyannote.audio.pipelines import MultiLabelSegmentation as MultiLabelSegmentationPipeline
from pyannote.database import registry, FileFinder
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio.models.segmentation import SSeRiouSS, PyanNet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        line = (
            f"SPEAKER {uri} {channel_id} "
            f"{start_time:.3f} {duration:.3f} "
            f"<NA> <NA> {speaker_id} <NA> <NA>\n"
        )
        file_handle.write(line)

def run_inference(model_name: str): # Add model_name argument
    try:
        # Define paths based on model_name
        checkpoint_path = f"outputs/model_checkpoints_{model_name}/mls_model_{model_name}.ckpt"
        optimized_params_path = f"outputs/configs_{model_name}/optimized_pipeline_params_{model_name}.json"
        results_dir = f"outputs/inference_results_{model_name}" # Make results directory model-specific
        os.makedirs(results_dir, exist_ok=True)

        # Check if model checkpoint and optimized parameters exist
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint not found at {checkpoint_path} for model {model_name}")
            return False
        if not os.path.exists(optimized_params_path):
            logger.error(f"Optimized parameters not found at {optimized_params_path} for model {model_name}")
            return False

        # Load dataset
        file_finder = FileFinder()
        registry.load_database("/home/nele_pauline_suffo/projects/pyannote-audio/project/data/database.yml")
        protocol = registry.get_protocol(
            'ChildLens.SpeakerDiarization.audio',
            preprocessors={"audio": lambda x: str(file_finder(x))}
        )
        logger.info("Loaded ChildLens dataset")

        # Load model - choose class based on model_name
        if model_name.lower() == "sseriouss":
            mls_model = SSeRiouSS.load_from_checkpoint(checkpoint_path)
        elif model_name.lower() == "pyannet":
            mls_model = PyanNet.load_from_checkpoint(checkpoint_path)
        else:
            logger.error(f"Unsupported model name: {model_name} for loading checkpoint.")
            return False
        logger.info(f"Loaded trained {model_name} model from checkpoint")

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
            # The pipeline returns an Annotation object
            speech_annotation = optimized_pipeline(file)
            # Set the URI for the annotation if the pipeline doesn't do it
            if not getattr(speech_annotation, 'uri', None):
                speech_annotation.uri = file_id

            _ = metric(file['annotation'], speech_annotation, uem=file['annotated'])
            output_path = os.path.join(results_dir, f"{file_id}.rttm")
            # Use write_rttm to save the Annotation object
            with open(output_path, "w") as f:
                write_rttm(f, speech_annotation)
            logger.info(f"Saved inference result for {file_id} to {output_path}")

        # Compute and log diarization error rate
        diarization_error_rate = abs(metric)
        logger.info(f"Diarization error rate for {model_name} = {diarization_error_rate * 100:.1f}%")
        der_summary_path = os.path.join(results_dir, f"diarization_error_rate_{model_name}.txt")
        with open(der_summary_path, "w") as f:
            f.write(f"Diarization error rate for {model_name} = {diarization_error_rate * 100:.1f}%")
        logger.info(f"Saved diarization error rate for {model_name} to {der_summary_path}")

        return True

    except Exception as e:
        logger.error(f"Error in run_inference for {model_name}: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a multi-label segmentation pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["sseriouss", "pyannet"],
        help="The model architecture whose pipeline to use for inference ('sseriouss' or 'pyannet')."
    )
    args = parser.parse_args()

    logger.info(f"Starting inference for model {args.model}")
    success = run_inference(model_name=args.model)
    if success:
        logger.info(f"Inference for model {args.model} completed successfully")
    else:
        logger.error(f"Inference for model {args.model} failed")