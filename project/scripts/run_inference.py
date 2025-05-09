import os
import json
import torch
from pyannote.audio import Inference
from pyannote.audio.pipelines import MultiLabelSegmentation as MultiLabelSegmentationPipeline
from pyannote.database import registry, FileFinder
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio.models.segmentation import SSeRiouSS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info("Initialized DetectionErrorRate metric")

        # Run inference and evaluation
        for file in protocol.test():
            file_id = file["uri"]
            speech = optimized_pipeline(file)
            _ = metric(file['annotation'], speech, uem=file['annotated'])
            output_path = os.path.join(results_dir, f"{file_id}.rttm")
            with open(output_path, "w") as f:
                speech.write(f)
            logger.info(f"Saved inference result for {file_id} to {output_path}")

        # Compute and log detection error rate
        detection_error_rate = abs(metric)
        logger.info(f"Detection error rate = {detection_error_rate * 100:.1f}%")
        with open(os.path.join(results_dir, "detection_error_rate.txt"), "w") as f:
            f.write(f"Detection error rate = {detection_error_rate * 100:.1f}%")
        logger.info("Saved detection error rate")

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