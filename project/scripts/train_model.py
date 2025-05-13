import os
import logging
import argparse
from pyannote.database import registry, FileFinder
from pyannote.audio.tasks import MultiLabelSegmentation
from pyannote.audio.models.segmentation import SSeRiouSS, PyanNet
from pytorch_lightning import Trainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(model_name: str):
    try:
        # Define output paths
        checkpoint_dir = f"outputs/model_checkpoints_{model_name}" # Make checkpoint_dir model-specific
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"mls_model_{model_name}.ckpt")

        # Load dataset
        file_finder = FileFinder()
        registry.load_database("/home/nele_pauline_suffo/projects/pyannote-audio/project/data/database.yml")
        protocol = registry.get_protocol(
            'ChildLens.SpeakerDiarization.audio',
            preprocessors={"audio": lambda x: str(file_finder(x))}
        )
        logger.info("Loaded ChildLens dataset")
        
        # Configure task
        mls_task = MultiLabelSegmentation(
            protocol,
            duration=2.0,
            batch_size=64,
            num_workers=12,
            classes=['KCHI', 'CHI', 'MAL', 'FEM', 'OVH', 'SPEECH'],
        )
        logger.info("Configured MultiLabelSegmentation task")

        # Initialize model based on the argument
        if model_name.lower() == "sseriouss":
            mls_model = SSeRiouSS(task=mls_task, wav2vec="WAVLM_BASE", linear={"hidden_size": 128, "num_layers": 3})
            logger.info("Initialized SSeRiouSS model")
        elif model_name.lower() == "pyannet":
            mls_model = PyanNet(task=mls_task, linear={"hidden_size": 128, "num_layers": 3})
            logger.info("Initialized PyanNet model")
        else:
            logger.error(f"Unsupported model name: {model_name}. Choose 'sseriouss' or 'pyannet'.")
            return False
        
        # Move model to GPU explicitly
        mls_model = mls_model.to("cuda:0")
        logger.info(f"Moved {model_name} model to cuda:0")
        
        print(f"Model device: {next(mls_model.parameters()).device}")
        
        # Configure trainer
        trainer = Trainer(
            devices=[0],
            accelerator="gpu",
            max_epochs=200,
            default_root_dir=checkpoint_dir,
            enable_checkpointing=True
        )
        logger.info("Configured PyTorch Lightning Trainer")

        # Train model
        trainer.fit(mls_model)
        logger.info(f"Model training completed for {model_name}")

        # Save final checkpoint
        trainer.save_checkpoint(checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

        return True

    except Exception as e:
        logger.error(f"Error in train_model for {model_name}: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-label segmentation model.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["sseriouss", "pyannet"],
        help="The model architecture to train ('sseriouss' or 'pyannet')."
    )
    args = parser.parse_args()

    logger.info(f"Starting model training for {args.model}")
    success = train_model(model_name=args.model)
    if success:
        logger.info(f"Model training for {args.model} completed successfully")
    else:
        logger.error(f"Model training for {args.model} failed")