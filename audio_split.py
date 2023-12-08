import os
import librosa
import concurrent.futures

import numpy as np
from tqdm import tqdm
import soundfile as sf
from audio_balancer import TrainingData


def split_audio_librosa(input_path: str, output_path: str, split_seconds: int = 300):
    files = []
    # Load the audio file using librosa
    data, rate = librosa.load(input_path, sr=None)

    # Convert to milliseconds and discard the first and last minute
    data = data[rate * 60:-rate * 60]

    # Split audio into chunks of length split_seconds
    for i in range(0, len(data), rate * split_seconds):
        chunk = data[i:i + rate * split_seconds]
        # Discard if chunk is less than split_seconds
        if len(chunk) == rate * split_seconds:
            if output_path is not None:
                # Save the result using soundfile
                file_name = output_path + f"({i // (rate * split_seconds)}).wav"
                files.append(file_name)
                if not os.path.exists(file_name):
                    sf.write(file_name, chunk, rate)

    return os.path.dirname(input_path).split('/')[-1], files


def split_audio_multi_thread(threads: int = 4, use_conf: bool = True, input_dir: str = None,
                             output_dir: str = None, split_seconds: int = 300,
                             selected: dict[str, TrainingData] = None) -> float:
    if use_conf:
        import configuration
        config = configuration.read_config()
        if not config.getboolean("Settings", "split files"):
            print("Splitting files is disabled in the configuration file, so this step has been skipped")
            return 0

        if config.getboolean("Settings", "remove silence"):
            input_dir = config["Paths"]["remove silence files"]
        elif config.getboolean("Settings", "Reduce Noise"):
            input_dir = config["Paths"]["denoised files"]
        elif config.getboolean("Settings", "Convert to wav"):
            input_dir = config["Paths"]["wav files"]
        else:
            input_dir = config["Paths"]["raw files"]

        output_dir = config["Paths"]["split files"]
        split_seconds = config.getint("Settings", "split seconds")

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    futures = []

    if not selected:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker_dir in os.listdir(input_dir):

                if not os.path.exists(os.path.join(output_dir, speaker_dir)):
                    os.makedirs(os.path.join(output_dir, speaker_dir))

                for file in os.listdir(os.path.join(input_dir, speaker_dir)):
                    if file.endswith(".wav"):
                        audio_path = os.path.join(os.path.join(input_dir, speaker_dir), file)
                        output_path = os.path.join(os.path.join(output_dir, speaker_dir), file[:-4] + "_split")

                        futures.append(executor.submit(split_audio_librosa, audio_path, output_path, split_seconds))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                               desc="Splitting audio files", colour="#4B0082"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred during splitting: {e}")

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker_dir in selected.keys():

                if not os.path.exists(os.path.join(output_dir, speaker_dir)):
                    os.makedirs(os.path.join(output_dir, speaker_dir))

                for file in selected.get(speaker_dir).Speaker_Files:
                    if file.endswith(".wav"):
                        audio_path = os.path.join(os.path.join(input_dir, speaker_dir), file)
                        output_path = os.path.join(os.path.join(output_dir, speaker_dir), file[:-4] + "_split")

                        futures.append(executor.submit(split_audio_librosa, audio_path, output_path, split_seconds))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                               desc="Splitting audio files", colour="#4B0082"):
                try:
                    speaker, file_names = future.result()
                    selected.get(speaker).Speaker_Files = file_names
                except Exception as e:
                    print(f"Exception occurred during splitting: {e}")
