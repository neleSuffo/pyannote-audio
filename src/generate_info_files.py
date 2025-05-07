import torchaudio
import torch
from pathlib import Path

def generate_info_files(lst_files, audio_dir, metadata_dir):
    """Generate torchaudio.info metadata files for audio files listed in lst_files.
    
    Parameters
    ----------
    lst_files : list of str
        Paths to .lst files (e.g., train.lst, dev.lst, test.lst).
    audio_dir : str or Path
        Directory containing .wav files.
    metadata_dir : str or Path
        Directory to store .info files.
    """
    audio_dir = Path(audio_dir)
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Collect all URIs from lst files
    uris = set()
    for lst_file in lst_files:
        with open(lst_file, 'r') as f:
            for line in f:
                uri = line.strip()
                if uri:
                    uris.add(uri)

    # Generate .info files for each URI
    for uri in uris:
        audio_path = audio_dir / f"{uri}.wav"
        info_path = metadata_dir / f"{uri}.info"
        if audio_path.exists():
            try:
                info = torchaudio.info(str(audio_path))
                torch.save(info, str(info_path))
                print(f"Generated {info_path}")
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
        else:
            print(f"Audio file not found: {audio_path}")

if __name__ == "__main__":
    audio_dir = "/home/nele_pauline_suffo/ProcessedData/childlens_audio"
    metadata_dir = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens/metadata"
    lst_files = [
        "/home/nele_pauline_suffo/ProcessedData/vtc_childlens/train.lst",
        "/home/nele_pauline_suffo/ProcessedData/vtc_childlens/dev.lst",
        "/home/nele_pauline_suffo/ProcessedData/vtc_childlens/test.lst"
    ]
    generate_info_files(lst_files, audio_dir, metadata_dir)