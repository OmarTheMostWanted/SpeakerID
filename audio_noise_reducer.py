import os
import concurrent.futures
from tqdm import tqdm
import noisereduce as nr
from scipy.io import wavfile
import warnings

from audio_balancer import TrainingData


def reduce_noise(input_path: str, output_path: str = None, device=None, chunk_size=100000):
    # Load the audio file
    rate, data = wavfile.read(input_path)

    if os.path.exists(output_path):
        return rate, data

    # Reduce noise, For AMD GPUs, you can use libraries like ROCm or numba with ROCm support.
    # Size of signal chunks to reduce noise over. Larger sizes will take more space in memory, smaller sizes can take
    # longer to compute.

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=1.0, chunk_size=chunk_size, device=device)

        if output_path is not None:
            # Save the result
            wavfile.write(output_path, rate, reduced_noise)

        return rate, reduced_noise


def reduce_noise_librosa(input_path: str, output_path: str = None, device=None, chunk_size=100000):
    import librosa
    import numpy as np

    # Load the audio file using librosa
    data, rate = librosa.load(input_path, sr=None)

    # Ensure the audio data is in the correct format for noise reduction
    data = np.asfortranarray(data)

    # Reduce noise
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=1.0, chunk_size=chunk_size,
                                    device=device)

    if output_path is not None:
        # Save the result using librosa
        librosa.output.write_wav(output_path, reduced_noise, rate)

    return reduced_noise


def reduce_noise_multi_thread(threads: int = 4, use_conf: bool = True, input_dir: str = None, out_put_dir: str = None,
                              device=None, chunk_size=100000, selected: dict[str, TrainingData] = None):
    if out_put_dir is not None and not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if use_conf:
        import configuration
        config = configuration.read_config()
        if not config.getboolean("Settings", "Reduce Noise"):
            print("Noise reduction is disabled in the configuration file, so this step has been skipped")
            return
        if config.getboolean("Settings", "balance"):
            input_dir = config["Paths"]["balanced files"]
        elif config.getboolean("Settings", "Convert to wav"):
            input_dir = config["Paths"]["wav files"]
        else:
            input_dir = config["Paths"]["raw files"]

        out_put_dir = config["Paths"]["denoised files"]
        device = config["Settings"]["device"]
        chunk_size = config.getint("Settings", "chunk size")

    elif input_dir is None or out_put_dir is None:
        raise Exception("Provide a directory or use config")

    futures = []

    if not selected:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker_dir in os.listdir(input_dir):

                if speaker_dir == "unused":
                    break

                for file in os.listdir(os.path.join(input_dir, speaker_dir)):
                    if file.endswith(".wav"):
                        audio_path = os.path.join(input_dir, speaker_dir, file)
                        denoised_path = os.path.join(out_put_dir, speaker_dir, file[:-4] + "_denoised.wav")
                        os.makedirs(os.path.dirname(denoised_path), exist_ok=True)

                        futures.append(executor.submit(reduce_noise, audio_path, denoised_path, device, chunk_size))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                               desc="Denoising audio files", colour="Blue"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred during denoising: {e}")

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker_name in selected.keys():

                new_files = []

                for file in selected.get(speaker_name).Speaker_Files:
                    if file.endswith(".wav"):
                        audio_path = os.path.join(input_dir, speaker_name, file)
                        denoised_path = os.path.join(out_put_dir, speaker_name, file[:-4] + "_denoised.wav")
                        new_files.append(os.path.basename(denoised_path))
                        os.makedirs(os.path.dirname(denoised_path), exist_ok=True)

                        futures.append(executor.submit(reduce_noise, audio_path, denoised_path, device, chunk_size))

                selected.get(speaker_name).Speaker_Files = new_files

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                               desc="Denoising audio files", colour="Blue"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred during denoising: {e}")
