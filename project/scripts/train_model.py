import os
from pyannote.database import registry, FileFinder
from pyannote.audio.tasks import MultiLabelSegmentation
from pyannote.audio.models.segmentation import SSeRiouSS
from pytorch_lightning import Trainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    try:
        # Define output paths
        checkpoint_dir = "outputs/model_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "mls_model.ckpt")

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
            batch_size=32,
            num_workers=4,
            classes=['kchi', 'och', 'mal', 'fem', 'ovh']
        )
        logger.info("Configured MultiLabelSegmentation task")

        # Initialize model
        mls_model = SSeRiouSS(task=mls_task, wav2vec="WAVLM_BASE")
        logger.info("Initialized SSeRiouSS model")

        # Configure trainer
        trainer = Trainer(
            devices=1,
            max_epochs=1,
            default_root_dir=checkpoint_dir,
            enable_checkpointing=True
        )
        logger.info("Configured PyTorch Lightning Trainer")

        # Train model
        trainer.fit(mls_model)
        logger.info("Model training completed")

        # Save final checkpoint
        trainer.save_checkpoint(checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

        return True

    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting model training")
    success = train_model()
    if success:
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed")