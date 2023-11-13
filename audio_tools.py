import os
import concurrent.futures
from pydub import AudioSegment
from tqdm import tqdm
import sys


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


def convert_to_wav_multi_thread(threads: int = 10, use_conf: bool = True, input_dir: str = None,
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
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ncols=70):
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
    print(f"Selected {len(selected_files)} files with total duration {selected_duration} seconds.")
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

    print(f"Target duration for each speaker: {speaker_with_min_duration.Speaker_Total_Duration} seconds.")

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

    speakers: [Speaker]

    speaker_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                    os.path.isdir(os.path.join(root_dir, d))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        speakers = list(executor.map(create_speaker, speaker_dirs))

    speaker_with_min_duration = min(speakers, key=lambda speaker: speaker.Speaker_Total_Duration)

    print(f"Target duration for each speaker: {speaker_with_min_duration.Speaker_Total_Duration} seconds.")

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

            for file in td.Speaker_Files:
                source = os.path.join(root_dir, td.Speaker_Name, file)
                dist = os.path.join(out_put_dir, td.Speaker_Name, file)
                shutil.copy(source, dist)

    return speaker_data


def normalize_audio_file(threads: int = 4, use_conf: bool = True, root_dir: str = None, out_put_dir: str = None,
                         target_amplitude=-20.0):

    if out_put_dir is not None and not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if use_conf:
        import configuration
        config = configuration.read_config()
        root_dir = config["Paths"]["balanced files"]
        out_put_dir = config["Paths"]["normalized files"]

    elif root_dir is None or out_put_dir is None:
        raise Exception("Provide a directory or use config")

    audio = AudioSegment.from_file(input_path)

    # Calculate the difference in dB between the target amplitude and the current amplitude
    diff = target_amplitude - audio.dBFS

    # Normalize the audio to the target amplitude
    normalized_audio = audio + diff

    # Export the normalized audio to the output path
    normalized_audio.export(output_path, format="wav")


if __name__ == '__main__':

    if len(sys.argv) == 1:
        training_data = balance_audio_multi_thread(use_conf=True)

    elif len(sys.argv) == 4 and sys.argv[1] == 'convert':
        convert_to_wav_multi_thread(input_dir=sys.argv[2], output_dir=sys.argv[3], use_conf=False)

    elif len(sys.argv) == 4 and sys.argv[1] == 'prepare':
        balance_audio_multi_thread(use_conf=False, root_dir=sys.argv[2], out_put_dir=sys.argv[3])

    else:
        print('Usage: AudioTools.py convert <input_dir> <output_dir>')
        print('Example: AudioTools.py convert ./input ./output')

        print('Usage: AudioTools.py prepare <input_dir> <output_dir>')
        print('Example: AudioTools.py prepare ./input ./output')
