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


def convert_to_wav(input_dir: str, output_dir: str, threads: int = 10) -> None:
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

    for file in tqdm(files, total=len(files), colour="green", dynamic_ncols=True,
                     desc=f"Calculating total audio duration for speaker {os.path.basename(audio_dir)}"):
        if file.endswith(".wav"):
            audio_file = AudioSegment.from_wav(os.path.join(audio_dir, file))
            total_duration += audio_file.duration_seconds
            audio_duration[audio_file] = total_duration

    return audio_duration, total_duration


def select_audio_files(speaker_dir: dict[str, float], target_duration: float) -> [str]:
    """Select audio files for a speaker to match the target duration."""
    print(f"Processing {speaker_dir}...")
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


def get_the_speaker_with_smallest_dataset(speakers_dir: str) -> (str, float):
    arr = []
    min = float("inf")
    speaker_min: str
    print(f"Getting the maximum number of audio seconds to use per speaker")
    for speaker in tqdm(os.listdir("audio_files_wav"), total=len(os.listdir("audio_files_wav")), colour="blue", dynamic_ncols=True):
        duration = get_audio_files_duration(os.path.join("audio_files_wav/", speaker))
        arr.append(duration)
        if duration < min:
            min = duration
            speaker_min = speaker
    print(f"The speaker with smallest data set is {speaker_min} with duration of {min}")
    return speaker_min, min


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
        self.Speaker_Name = os.path.dirname(speaker_dir_path)
        self.Speaker_Files = speaker_files


def prepare_training_data():
    import configuration

    config = configuration.read_config()

    root_dir = config["Paths"]["training data"]

    speakers: [Speaker] = []

    speaker_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                    os.path.isdir(os.path.join(root_dir, d))]

    for speaker_dir in speaker_dirs:
        speakers.append(Speaker(speaker_dir))

    speaker_with_min_duration = min(speakers, key=lambda speaker: speaker.Speaker_Total_Duration)

    print(f"Target duration for each speaker: {speaker_with_min_duration.Speaker_Total_Duration} seconds.")

    speaker_files = [TrainingData(speaker.Speaker_Path, select_audio_files(speaker.Speaker_Audio_Files, speaker_with_min_duration.Speaker_Total_Duration)) for
                     speaker in
                     tqdm(speakers, total=len(speaker_dirs), dynamic_ncols=True, colour="blue", desc="Selecting audio files to balance the training data")]

    print("Selection completed.")
    return speaker_files

def prepare_training_data_multi_thread(threads = 4):
    import configuration

    config = configuration.read_config()

    root_dir = config["Paths"]["training data"]

    def create_speaker(speaker_dir):
        return Speaker(speaker_dir)

    speakers: [Speaker] = []

    speaker_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                    os.path.isdir(os.path.join(root_dir, d))]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        speakers = list(executor.map(create_speaker, speaker_dirs))

    speaker_with_min_duration = min(speakers, key=lambda speaker: speaker.Speaker_Total_Duration)

    print(f"Target duration for each speaker: {speaker_with_min_duration.Speaker_Total_Duration} seconds.")

    speaker_files = [TrainingData(speaker.Speaker_Path, select_audio_files(speaker.Speaker_Audio_Files, speaker_with_min_duration.Speaker_Total_Duration)) for
                     speaker in
                     tqdm(speakers, total=len(speaker_dirs), dynamic_ncols=True, colour="blue", desc="Selecting audio files to balance the training data")]

    print("Selection completed.")
    return speaker_files


if __name__ == '__main__':
    if len(sys.argv) == 1:
        prepare_training_data_multi_thread()

    if len(sys.argv) == 4 and sys.argv[1] == 'convert':
        convert_to_wav(sys.argv[2], sys.argv[3])
    else:
        print('Usage: AudioTools.py convert <input_dir> <output_dir>')
        print('Example: AudioTools.py convert ./input ./output')
