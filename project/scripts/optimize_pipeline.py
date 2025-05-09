import os
import json
from pyannote.database import registry, FileFinder
from pyannote.audio.pipelines import MultiLabelSegmentation as MultiLabelSegmentationPipeline
from pyannote.pipeline import Optimizer
from pyannote.audio.models.segmentation import SSeRiouSS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_pipeline():
    try:
        # Define paths
        checkpoint_path = "outputs/model_checkpoints/mls_model.ckpt"
        output_dir = "outputs/configs"
        os.makedirs(output_dir, exist_ok=True)
        optimized_params_path = os.path.join(output_dir, "optimized_pipeline_params.json")

        # Check if model checkpoint exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint not found at {checkpoint_path}")
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

        # Define initial parameters
        initial_params = {
            "thresholds": {
                "KCHI": {"onset": 0.6, "offset": 0.4},
                "OCH": {"onset": 0.6, "offset": 0.4},
                "FEM": {"onset": 0.6, "offset": 0.4},
                "MAL": {"onset": 0.6, "offset": 0.4},
                "SPEECH": {"onset": 0.6, "offset": 0.4},
            },
            "min_duration_on": 0.0,
            "min_duration_off": 0.0,
        }
        pipeline.instantiate(initial_params)
        logger.info("Instantiated pipeline with initial parameters")

        # Freeze non-optimized parameters
        pipeline.freeze({'min_duration_on': 0.0, 'min_duration_off': 0.0})
        logger.info("Froze min_duration_on and min_duration_off")

        # Optimize pipeline parameters
        optimizer = Optimizer(pipeline)
        optimizer.tune(
            list(protocol.development()),
            warm_start=initial_params,
            n_iterations=20,
            show_progress=False
        )
        optimized_params = optimizer.best_params
        logger.info(f"Optimized pipeline parameters: {optimized_params}")

        # Save optimized parameters
        with open(optimized_params_path, 'w') as f:
            json.dump(optimized_params, f, indent=4)
        logger.info(f"Saved optimized pipeline parameters to {optimized_params_path}")

        return True

    except Exception as e:
        logger.error(f"Error in optimize_pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting pipeline parameter optimization")
    success = optimize_pipeline()
    if success:
        logger.info("Pipeline parameter optimization completed successfully")
    else:
        logger.error("Pipeline parameter optimization failed")