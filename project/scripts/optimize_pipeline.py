import os
import json
import argparse # Import argparse
from pyannote.database import registry, FileFinder
from pyannote.audio.pipelines import MultiLabelSegmentation as MultiLabelSegmentationPipeline
from pyannote.pipeline import Optimizer
from pyannote.audio.models.segmentation import SSeRiouSS, PyanNet
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_pipeline(model_name: str):
    try:
        # Define paths based on model_name
        checkpoint_path = f"outputs/model_checkpoints_{model_name}/mls_model_{model_name}.ckpt"
        output_dir = f"outputs/configs_{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        optimized_params_path = os.path.join(output_dir, f"optimized_pipeline_params_{model_name}.json")

        # Check if model checkpoint exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"Model checkpoint not found at {checkpoint_path} for model {model_name}")
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

        # Define initial parameters
        initial_params = {
            "thresholds": {
                "KCHI": {"onset": 0.6, "offset": 0.4},
                "CHI": {"onset": 0.6, "offset": 0.4},
                "FEM": {"onset": 0.6, "offset": 0.4},
                "MAL": {"onset": 0.6, "offset": 0.4},
                "OVH": {"onset": 0.6, "offset": 0.4},
                "SPEECH": {"onset": 0.5, "offset": 0.5},
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
            n_iterations=40,
            show_progress=False
        )
        optimized_params = optimizer.best_params
        logger.info(f"Optimized pipeline parameters for {model_name}: {optimized_params}")

        # Save optimized parameters
        with open(optimized_params_path, 'w') as f:
            json.dump(optimized_params, f, indent=4)
        logger.info(f"Saved optimized pipeline parameters for {model_name} to {optimized_params_path}")

        return True

    except Exception as e:
        logger.error(f"Error in optimize_pipeline for {model_name}: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize a multi-label segmentation pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["sseriouss", "pyannet"],
        help="The model architecture for which to optimize the pipeline ('sseriouss' or 'pyannet')."
    )
    args = parser.parse_args()

    logger.info(f"Starting pipeline parameter optimization for model {args.model}")
    success = optimize_pipeline(model_name=args.model)
    if success:
        logger.info(f"Pipeline parameter optimization for model {args.model} completed successfully")
    else:
        logger.error(f"Pipeline parameter optimization for model {args.model} failed")