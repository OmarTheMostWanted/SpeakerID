import os
import concurrent.futures

import numpy
from pydub import AudioSegment
from tqdm import tqdm
import sys
import noisereduce as nr
from scipy.io import wavfile


def convert_file_to_wav(audio_path: str, wav_path: str, audio_format: str) -> None:
    # Convert audio to wav
    if audio_format == 'mp3':
        audio = AudioSegment.from_mp3(audio_path)
    elif audio_format == 'm4a':
        audio = AudioSegment.from_file(audio_path, format='m4a')
    else:
        print("Unsupported format")
        return
    audio.export(wav_path, format='wav')


def convert_to_wav_multi_thread(threads: int = 4, use_conf: bool = True, input_dir: str = None,
                                output_dir: str = None) -> None:
    if use_conf:
        import configuration
        config = configuration.read_config()
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
                if file.endswith('.mp3') or file.endswith('.m4a'):
                    audio_path = os.path.join(root, file)
                    rel_root = os.path.relpath(root, input_dir)
                    wav_path = os.path.join(output_dir, rel_root, file[:-4] + '.wav')

                    # Create new directories in output_dir as necessary
                    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

                    # Submit a new task to the thread pool
                    if file.endswith('.mp3'):
                        futures.append(executor.submit(convert_file_to_wav, audio_path, wav_path, 'mp3'))
                    elif file.endswith('.m4a'):
                        futures.append(executor.submit(convert_file_to_wav, audio_path, wav_path, 'm4a'))

        # Add progress bar using tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True, desc="Converting files to wav"):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred during conversion: {e}")


def get_audio_files_duration(audio_dir: str) -> (dict[str, float], float):
    """Calculate the total duration of audio files in seconds."""
    audio_duration: dict[str, float] = dict()
    total_duration = 0.0

    files = os.listdir(audio_dir)

    for file in files:
        if file.endswith(".wav"):
            audio_file = AudioSegment.from_wav(os.path.join(audio_dir, file))
            audio_file_duration = audio_file.duration_seconds
            total_duration += audio_file_duration
            audio_duration[file] = audio_file_duration
    return audio_duration, total_duration


def select_audio_files(speaker_dir: dict[str, float], target_duration: float) -> [str]:
    """Select audio files for a speaker to match the target duration."""
    sorted_map = dict(sorted(speaker_dir.items(), key=lambda item: item[1], reverse=True))
    selected_files = []
    selected_duration = 0.0
    for audio_file, file_duration in sorted_map.items():
        if selected_duration + file_duration <= target_duration:
            selected_files.append(audio_file)
            selected_duration += file_duration

        if selected_duration > target_duration:
            break
    # print(f"Selected {len(selected_files)} files with total duration {selected_duration} seconds.")
    return selected_files


class Speaker:
    Speaker_Path: str
    Speaker_Total_Duration: float
    Speaker_Audio_Files: dict[str, float]

    def __init__(self, speaker_path):
        # Instance variables
        self.Speaker_Path = speaker_path
        self.Speaker_Audio_Files, self.Speaker_Total_Duration = get_audio_files_duration(speaker_path)


class TrainingData:
    Speaker_Name: str
    Speaker_Files: [str]

    def __init__(self, speaker_dir_path, speaker_files):
        # Instance variables
        self.Speaker_Name = os.path.basename(speaker_dir_path)
        self.Speaker_Files = speaker_files


def balance_audio():
    import configuration

    config = configuration.read_config()

    root_dir = config["Paths"]["training data"]

    speakers: [Speaker] = []

    speaker_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                    os.path.isdir(os.path.join(root_dir, d))]

    for speaker_dir in tqdm(speaker_dirs, colour="green", dynamic_ncols=True,
                            desc="Calculating total audio duration for each speaker."):
        speakers.append(Speaker(speaker_dir))

    speaker_with_min_duration = min(speakers, key=lambda speaker: speaker.Speaker_Total_Duration)

    print(f"Target duration for each speaker: {round(speaker_with_min_duration.Speaker_Total_Duration, 1)} seconds.")

    speaker_data: [TrainingData] = []

    for speaker in tqdm(speakers, colour="blue", dynamic_ncols=True, desc="Balancing audio data for all speakers."):
        selected_files = select_audio_files(speaker.Speaker_Audio_Files,
                                            speaker_with_min_duration.Speaker_Total_Duration)
        speaker_data.append(TrainingData(speaker.Speaker_Path, selected_files))

    print("Selection completed.")
    return speaker_data


def balance_audio_multi_thread(threads: int = 4, use_conf: bool = True, root_dir: str = None,
                               out_put_dir: str = None) -> [TrainingData]:
    if out_put_dir is not None and not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if use_conf:
        import configuration
        config = configuration.read_config()
        root_dir = config["Paths"]["wav files"]
        out_put_dir = config["Paths"]["balanced files"]

    elif root_dir is None or out_put_dir is None:
        raise Exception("Provide a directory or use config")

    def create_speaker(speaker_dir: str) -> Speaker:
        return Speaker(speaker_dir)

    speakers: [Speaker] = []

    speaker_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                    os.path.isdir(os.path.join(root_dir, d))]

    speaker_futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:

        for speaker_dir in speaker_dirs:
            speaker_futures.append(executor.submit(create_speaker, speaker_dir))
        # speakers = list(executor.map(create_speaker, speaker_dirs))
        for future in tqdm(concurrent.futures.as_completed(speaker_futures), total=len(speaker_futures), colour="MAGENTA", dynamic_ncols=True,
                           desc=f"Balancing audio files"):
            try:
                speakers.append(future.result())
            except Exception as e:
                print(f"Exception occurred during copying: {e}")

    speaker_with_min_duration = min(speakers, key=lambda speaker: speaker.Speaker_Total_Duration)

    print(f"Target duration for each speaker: {round(speaker_with_min_duration.Speaker_Total_Duration, 2)} seconds.")

    def create_training_data(s: Speaker) -> TrainingData:
        sf = select_audio_files(s.Speaker_Audio_Files,
                                speaker_with_min_duration.Speaker_Total_Duration)
        return TrainingData(s.Speaker_Path, sf)

    speaker_data: [TrainingData]

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        speaker_data = list(executor.map(create_training_data, speakers))

    res = []
    for s in speaker_data:
        res.append((s.Speaker_Name, len(s.Speaker_Files)))

    for re in res:
        print(f"Selection completed: {re}")

    if out_put_dir is not None:
        if not os.path.exists(out_put_dir):
            os.makedirs(out_put_dir)

        import shutil
        for td in speaker_data:

            if not os.path.exists(os.path.join(out_put_dir, td.Speaker_Name)):
                os.makedirs(os.path.join(out_put_dir, td.Speaker_Name))

            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                speaker_data = list(executor.map(create_training_data, speakers))
                for file in td.Speaker_Files:
                    source = os.path.join(root_dir, td.Speaker_Name, file)
                    dist = os.path.join(out_put_dir, td.Speaker_Name, file[:-4] + "_balanced.wav")
                    futures.append(executor.submit(shutil.copy, source, dist))

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), colour="32cd32", dynamic_ncols=True,
                                   desc=f"Copying balanced audio files of {td.Speaker_Name}"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Exception occurred during copying: {e}")

    return speaker_data


def normalize_file(input_path: str, out_put_dir: str, target_amplitude: float) -> None:
    audio = AudioSegment.from_file(input_path)

    # Calculate the difference in dB between the target amplitude and the current amplitude
    diff = target_amplitude - audio.dBFS

    # Normalize the audio to the target amplitude
    normalized_audio = audio + diff

    # Check for clipping
    if normalized_audio.max_dBFS > 0:
        print(f"Warning: Clipping occurred in {input_path}")

    # Export the normalized audio to the output path
    normalized_audio.export(out_put_dir, format="wav")


def calculate_average_amplitude(directory: str) -> float:
    total_amplitude = 0.0
    file_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):  # or whatever file extension you're using
            audio = AudioSegment.from_file(os.path.join(directory, filename))
            total_amplitude += audio.dBFS
            file_count += 1

    return total_amplitude / file_count


def normalize_audio_files_multi_thread(threads: int = 4, use_conf: bool = True, input_path: str = None, out_put_dir: str = None,
                                       target_amplitude=20.0, use_average_amplitude: bool = False):
    if out_put_dir is not None and not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if use_conf:
        import configuration
        config = configuration.read_config()
        input_path = config["Paths"]["balanced files"]
        out_put_dir = config["Paths"]["normalized files"]

    elif input_path is None or out_put_dir is None:
        raise Exception("Provide a directory or use config")

    if use_average_amplitude:
        amplitude_futures = []

        amplitudes: [float] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker_dir in os.listdir(input_path):
                amplitude_futures.append(executor.submit(calculate_average_amplitude, os.path.join(input_path, speaker_dir)))

            for future in tqdm(concurrent.futures.as_completed(amplitude_futures), total=len(amplitude_futures), dynamic_ncols=True,
                               desc="Calculating average amplitude", colour="red"):
                try:
                    amplitudes.append(future.result())
                except Exception as e:
                    print(f"Exception occurred during amplitude calculation {e}")

        target_amplitude = numpy.mean(amplitudes)

    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for speaker_dir in os.listdir(input_path):
            for file in os.listdir(os.path.join(input_path, speaker_dir)):
                if file.endswith(".wav"):
                    audio_path = os.path.join(input_path, speaker_dir, file)
                    normalized_path = os.path.join(out_put_dir, speaker_dir, file[:-4] + f"_normalized({round(target_amplitude, 1)}).wav")
                    os.makedirs(os.path.dirname(normalized_path), exist_ok=True)

                    futures.append(executor.submit(normalize_file, audio_path, normalized_path, target_amplitude))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                           desc=f"Normalizing audio files to amplitude {round(target_amplitude, 1)} ", colour="CYAN"):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred during normalizing: {e}")


def reduce_noise(input_path: str, output_path: str) -> None:
    # Load the audio file
    rate, data = wavfile.read(input_path)

    # Reduce noise
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    # Save the result
    wavfile.write(output_path, rate, reduced_noise)


def reduce_noise_multi_thread(threads: int = 4, use_conf: bool = True, input_path: str = None, out_put_dir: str = None):
    if out_put_dir is not None and not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if use_conf:
        import configuration
        config = configuration.read_config()
        input_path = config["Paths"]["normalized files"]
        out_put_dir = config["Paths"]["deionised files"]

    elif input_path is None or out_put_dir is None:
        raise Exception("Provide a directory or use config")

    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for speaker_dir in os.listdir(input_path):
            for file in os.listdir(os.path.join(input_path, speaker_dir)):
                if file.endswith(".wav"):
                    audio_path = os.path.join(input_path, speaker_dir, file)
                    denoised_path = os.path.join(out_put_dir, speaker_dir, file[:-4] + "_denoised.wav")
                    os.makedirs(os.path.dirname(denoised_path), exist_ok=True)

                    futures.append(executor.submit(reduce_noise, audio_path, denoised_path))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True, desc="Denoising audio files", colour="Blue"):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred during denoising: {e}")


if __name__ == '__main__':

    if len(sys.argv) == 1:
        convert_to_wav_multi_thread(use_conf=True, threads=15)
        training_data = balance_audio_multi_thread(use_conf=True, threads=15)
        normalize_audio_files_multi_thread(use_conf=True, threads=15, use_average_amplitude=True)
        reduce_noise_multi_thread(threads=15, use_conf=True)

    elif len(sys.argv) == 4 and sys.argv[1] == 'convert':
        convert_to_wav_multi_thread(input_dir=sys.argv[2], output_dir=sys.argv[3], use_conf=False)

    elif len(sys.argv) == 4 and sys.argv[1] == 'prepare':
        balance_audio_multi_thread(use_conf=False, root_dir=sys.argv[2], out_put_dir=sys.argv[3])

    else:
        print('Usage: AudioTools.py convert <input_dir> <output_dir>')
        print('Example: AudioTools.py convert ./input ./output')

        print('Usage: AudioTools.py prepare <input_dir> <output_dir>')
        print('Example: AudioTools.py prepare ./input ./output')
