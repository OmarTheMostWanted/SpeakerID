import concurrent.futures
import warnings

from pydub import AudioSegment
from tqdm import tqdm
import os


def convert_file_to_wav(audio_path: str, wav_path: str, replace: bool = False) -> None:
    if not replace and os.path.exists(wav_path):
        return

    # Convert audio to wav
    try:
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format='wav')
    except Exception as e:
        m = f"Error reading audio file: {os.path.basename(audio_path)}"
        warnings.warn(m)


def convert_to_wav_multi_thread(threads: int = 4, use_conf: bool = True, input_dir: str = None,
                                output_dir: str = None) -> None:
    if use_conf:
        import configuration
        config = configuration.read_config()
        if not config.getboolean("Settings", "Convert to wav"):
            print("Converting to wav is disabled in the configuration file, so this step has been skipped")
            return

        input_dir = config["Paths"]["raw files"]
        output_dir = config["Paths"]["wav files"]

    elif input_dir is None or output_dir is None:
        raise Exception("Provide a directory or use config")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                audio_path = os.path.join(root, file)
                rel_root = os.path.relpath(root, input_dir)
                wav_path = os.path.join(output_dir, rel_root, file[:-4] + '.wav')

                # Create new directories in output_dir as necessary
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)

                # Submit a new task to the thread pool
                futures.append(executor.submit(convert_file_to_wav, audio_path, wav_path, False))

        # Add progress bar using tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                           desc="Converting files to wav" , colour="white"):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred during conversion: {e}")
