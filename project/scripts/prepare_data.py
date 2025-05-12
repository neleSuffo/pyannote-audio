import os
import json
from pyannote.database import registry, FileFinder
from pyannote.audio.tasks import MultiLabelSegmentation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data():
    try:
        # Initialize file finder
        file_finder = FileFinder()
        logger.info("Initialized FileFinder")

        # Load dataset
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
            pin_memory=True,
        )
        logger.info("Configured MultiLabelSegmentation task")

        return True

    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting data preparation")
    success = prepare_data()
    if success:
        logger.info("Data preparation completed successfully")
    else:
        logger.error("Data preparation failed")