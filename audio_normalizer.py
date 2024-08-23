import os
import concurrent.futures
import warnings
import numpy
from pydub import AudioSegment
from tqdm import tqdm
from audio_balancer import TrainingData
import librosa
import numpy as np

def calculate_parameters(file_paths):
    # Initialize variables to calculate mean and standard deviation
    total_sum = 0
    total_sq_sum = 0
    count = 0

    for file_path in file_paths:
        # Load the audio file with librosa
        samples, _ = librosa.load(file_path, sr=None)

        # Update the sums and count
        total_sum += np.sum(samples)
        total_sq_sum += np.sum(samples ** 2)
        count += len(samples)

    # Calculate the mean and standard deviation
    mean = total_sum / count
    std_dev = np.sqrt((total_sq_sum / count) - (mean ** 2))

    return mean, std_dev


def normalize_file(input_path, mean, std_dev):
    # Load the audio file with librosa
    samples, sr = librosa.load(input_path, sr=None)

    # Normalize the samples
    normalized_samples = (samples - mean) / std_dev

    return normalized_samples, sr


def normalize_file(input_path: str, out_put_dir: str = None, target_amplitude: float = -20):
    audio = AudioSegment.from_file(input_path)

    if os.path.exists(out_put_dir):
        return audio

    # Calculate the difference in dB between the target amplitude and the current amplitude
    diff = target_amplitude - audio.dBFS

    # Normalize the audio to the target amplitude
    normalized_audio = audio + diff

    # Check for clipping
    if normalized_audio.max_dBFS > 0:
        print(f"Warning: Clipping occurred in {input_path}")

    if out_put_dir is not None:
        # Export the normalized audio to the output path
        normalized_audio.export(out_put_dir, format="wav")

    return normalized_audio


def normalize_file_librosa(input_path: str, output_dir: str = None, target_amplitude: float = -20.0):
    if os.path.exists(output_dir):
        return

    # Load the audio file with librosa
    y, sr = librosa.load(input_path, sr=None)

    # Calculate the current amplitude (in dB)
    current_amplitude = 20 * np.log10(np.max(np.abs(y)))

    # Calculate the difference in dB between the target amplitude and the current amplitude
    diff = target_amplitude - current_amplitude

    # Normalize the audio to the target amplitude
    normalized_audio = librosa.util.normalize(y) * (10 ** (diff / 20.0))

    # Check for clipping
    if np.max(np.abs(normalized_audio)) > 1:
        print(f"Warning: Clipping occurred in {input_path}")

    if output_dir is not None:
        # Export the normalized audio to the output path
        librosa.output.write_wav(output_dir, normalized_audio, sr)


def calculate_average_amplitude(directory: str) -> float:
    total_amplitude = 0.0
    file_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):  # or whatever file extension you're using
            audio = AudioSegment.from_file(os.path.join(directory, filename))
            total_amplitude += audio.dBFS
            file_count += 1

    if file_count == 0:
        warnings.warn(f"File count was 0 while calculating average amplitude in directory {directory}")
        return 0
    return total_amplitude / file_count


def normalize_audio_files_multi_thread(threads: int = 4, use_conf: bool = True, input_dir: str = None,
                                       out_put_dir: str = None,
                                       target_amplitude=20.0, use_average_amplitude: bool = False,
                                       selected: dict[str, TrainingData] = None) -> float:
    if use_conf:
        import configuration
        config = configuration.read_config()
        if not config.getboolean("Settings", "Normalize"):
            print("Normalizing is disabled in the configuration file, so this step has been skipped")
            return 0

        if config.getboolean("Settings", "balance"):
            input_dir = config["Paths"]["balanced files"]
        elif config.getboolean("Settings", "Split files"):
            input_dir = config["Paths"]["Split Files"]
        elif config.getboolean("Settings", "remove silence"):
            input_dir = config["Paths"]["remove silence files"]
        elif config.getboolean("Settings", "Reduce Noise"):
            input_dir = config["Paths"]["denoised files"]
        elif config.getboolean("Settings", "Convert to wav"):
            input_dir = config["Paths"]["wav files"]
        else:
            input_dir = config["Paths"]["raw files"]

        out_put_dir = config["Paths"]["normalized files"]
        use_average_amplitude = config.getboolean("Settings", "average amplitude")
        target_amplitude = config.getfloat("Settings", "target amplitude")

        if use_average_amplitude and target_amplitude and target_amplitude != 0:
            warnings.warn(
                "Both average amplitude and target amplitude are present in the config, choosing average amplitude")

    elif input_dir is None or out_put_dir is None:
        raise Exception("Provide a directory or use config")

    if out_put_dir is not None and not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if use_average_amplitude:
        amplitude_futures = []

        amplitudes: [float] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker_dir in os.listdir(input_dir):

                if speaker_dir == "unused":
                    continue

                amplitude_futures.append(
                    executor.submit(calculate_average_amplitude, os.path.join(input_dir, speaker_dir)))

            for future in tqdm(concurrent.futures.as_completed(amplitude_futures), total=len(amplitude_futures),
                               dynamic_ncols=True,
                               desc="Calculating average amplitude", colour="red"):
                try:
                    amplitudes.append(future.result())
                except Exception as e:
                    print(f"Exception occurred during amplitude calculation {e}")

        target_amplitude = numpy.mean(amplitudes)

    futures = []

    if not selected:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker_dir in os.listdir(input_dir):

                if speaker_dir == "unused":
                    continue

                for file in os.listdir(os.path.join(input_dir, speaker_dir)):
                    if file.endswith(".wav"):
                        audio_path = os.path.join(input_dir, speaker_dir, file)
                        normalized_path = os.path.join(out_put_dir, speaker_dir,
                                                       file[:-4] + f"_normalized({round(target_amplitude, 2)}).wav")
                        os.makedirs(os.path.dirname(normalized_path), exist_ok=True)

                        futures.append(executor.submit(normalize_file, audio_path, normalized_path, target_amplitude))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                               desc=f"Normalizing audio files to amplitude {round(target_amplitude, 1)} ",
                               colour="CYAN"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred during normalizing: {e}")

        return target_amplitude

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker_name in selected.keys():

                new_files = []

                for file in selected.get(speaker_name).Speaker_Files:
                    if file.endswith(".wav"):
                        audio_path = os.path.join(input_dir, speaker_name, file)
                        normalized_path = os.path.join(out_put_dir, speaker_name,
                                                       file[:-4] + f"_normalized({round(target_amplitude, 2)}).wav")

                        new_files.append(os.path.basename(normalized_path))

                        os.makedirs(os.path.dirname(normalized_path), exist_ok=True)

                        futures.append(executor.submit(normalize_file, audio_path, normalized_path, target_amplitude))

                selected.get(speaker_name).Speaker_Files = new_files

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                               desc=f"Normalizing audio files to amplitude {round(target_amplitude, 1)} ",
                               colour="CYAN"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception occurred during normalizing: {e}")

        return target_amplitude
