import os
import concurrent.futures
import warnings

from pydub import AudioSegment
from tqdm import tqdm

def convert_seconds(total_seconds):
    # Define the units of time and their lengths in seconds
    time_units = {
        "years": 60*60*24*365,
        "months": 60*60*24*30,
        "weeks": 60*60*24*7,
        "days": 60*60*24,
        "hours": 60*60,
        "minutes": 60,
        "seconds": 1
    }

    # Initialize the result dictionary
    result = {}

    # Iterate over the time units from largest to smallest
    for unit in time_units:
        # Calculate the number of full units that fit into the total seconds
        value = total_seconds // time_units[unit]
        # Update the total seconds with the remainder
        total_seconds %= time_units[unit]
        # Include the unit in the result if its value is not zero or if there is a unit to the left of it
        if value > 0 or result:
            result[unit] = value

    # Convert the result dictionary into a string
    result_str = ', '.join(f"{value} {unit}" for unit, value in result.items())

    # Return the result string
    return result_str

class Speaker:
    Speaker_Path: str
    Speaker_Total_Duration: float
    Speaker_Audio_Files: dict[str, float]

    def __init__(self, speaker_path):
        self.Speaker_Path = speaker_path
        self.Speaker_Audio_Files, self.Speaker_Total_Duration = get_audio_files_duration(speaker_path)


class TrainingData:
    Speaker_Name: str
    Speaker_Files: [str]
    Unused_Files: [str]

    def __init__(self, speaker_dir_path, speaker_files, unused_files):
        self.Speaker_Name = os.path.basename(speaker_dir_path)
        self.Speaker_Files = speaker_files
        self.Unused_Files = unused_files


def get_audio_files_duration(audio_dir: str) -> (dict[str, float], float):
    """Calculate the total duration of audio files in seconds."""
    audio_duration: dict[str, float] = dict()
    total_duration = 0.0

    files = os.listdir(audio_dir)

    for file in files:
        if file.endswith(".wav") or file.endswith(".WAV"):
            audio_file = AudioSegment.from_wav(os.path.join(audio_dir, file))
            audio_file_duration = audio_file.duration_seconds
            total_duration += audio_file_duration
            audio_duration[file] = audio_file_duration
        else:
            warnings.warn(f"{file} is not wav and it has been ignored")
    return audio_duration, total_duration


def select_audio_files(speaker_dir: dict[str, float], target_duration: float) -> ([str], [str]):
    """Select audio files for a speaker to match the target duration."""
    sorted_map = dict(sorted(speaker_dir.items(), key=lambda item: item[1], reverse=True))
    selected_files = []
    unused_files = []
    selected_duration = 0.0
    for audio_file, file_duration in sorted_map.items():
        if selected_duration + file_duration <= target_duration:
            selected_files.append(audio_file[:-4] + "_balanced.wav")
            selected_duration += file_duration
        else:
            unused_files.append(audio_file[:-4] + "_unused.wav")
    return selected_files, unused_files


def balance_audio_multi_thread(threads: int = 4, use_conf: bool = True, input_dir: str = None,
                               out_put_dir: str = None, target_duration=-1) -> dict[str, TrainingData]:
    if out_put_dir is not None and not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if use_conf:
        import configuration
        config = configuration.read_config()
        if not config.getboolean("Settings", "Balance"):
            print("Balancing is disabled in the configuration file, so this step has been skipped")
            return dict()
        if config.getboolean("Settings", "Split files"):
            input_dir = config["Paths"]["Split Files"]
        elif config.getboolean("Settings", "Remove Silence"):
            input_dir = config["Paths"]["remove silence files"]
        elif config.getboolean("Settings", "Reduce Noise"):
            input_dir = config["Paths"]["denoised files"]
        elif config.getboolean("Settings", "Convert to wav"):
            input_dir = config["Paths"]["wav files"]
        else:
            input_dir = config["Paths"]["raw files"]

        out_put_dir = config["Paths"]["balanced files"]
        target_duration = config.getfloat("Settings", "audio duration")

    elif input_dir is None or out_put_dir is None:
        raise Exception("Provide a directory or use config")

    def create_speaker(speaker_dir_a: str) -> Speaker:
        return Speaker(speaker_dir_a)

    speakers: [Speaker] = []

    speaker_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if
                    os.path.isdir(os.path.join(input_dir, d))]

    speaker_futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:

        for speaker_dir in speaker_dirs:
            speaker_futures.append(executor.submit(create_speaker, speaker_dir))
        # speakers = list(executor.map(create_speaker, speaker_dirs))
        for future in tqdm(concurrent.futures.as_completed(speaker_futures), total=len(speaker_futures),
                           colour="#ff69b4", dynamic_ncols=True,
                           desc=f"Balancing audio files"):
            try:
                speakers.append(future.result())
            except Exception as e:
                print(f"Exception occurred during copying: {e}")

    speaker_with_min_duration = min(speakers, key=lambda speaker: speaker.Speaker_Total_Duration)
    min_duration = speaker_with_min_duration.Speaker_Total_Duration

    if target_duration != -1 and min_duration < target_duration:
        warnings.warn("The target duration provided was bigger that the speaker with the largest audio duration, using the minimum duration instead")

    if target_duration != -1 and min_duration > target_duration:
        min_duration = target_duration

    print(f"Target duration for each speaker: {convert_seconds(round(min_duration, 2))}.")

    def create_training_data(speaker_a: Speaker) -> TrainingData:
        sf, unused = select_audio_files(speaker_a.Speaker_Audio_Files, min_duration)
        return TrainingData(speaker_a.Speaker_Path, sf, unused)

    speaker_data: [TrainingData]

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        speaker_data = list(executor.map(create_training_data, speakers))

    res = []
    for s in speaker_data:
        res.append((s.Speaker_Name, len(s.Speaker_Files)))

    if out_put_dir is not None:
        if not os.path.exists(out_put_dir):
            os.makedirs(out_put_dir)

        import shutil
        for td in speaker_data:

            if not os.path.exists(os.path.join(out_put_dir, td.Speaker_Name)):
                os.makedirs(os.path.join(out_put_dir, td.Speaker_Name))

            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for file in td.Speaker_Files:
                    source: str = os.path.join(input_dir, td.Speaker_Name, file[:-13] + ".wav")
                    dist: str = os.path.join(out_put_dir, td.Speaker_Name, file)
                    if not os.path.exists(dist):
                        futures.append(executor.submit(shutil.copy2, source, dist))

                os.makedirs(os.path.join(out_put_dir, "unused", td.Speaker_Name), exist_ok=True)

                for file in td.Unused_Files:
                    source = os.path.join(input_dir, td.Speaker_Name, file[:-11] + ".wav")
                    dist = os.path.join(out_put_dir, "unused", td.Speaker_Name, file)
                    if not os.path.exists(dist):
                        futures.append(executor.submit(shutil.copy2, source, dist))

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), colour="#32cd32",
                                   dynamic_ncols=True, desc=f"Copying balanced audio files of {td.Speaker_Name}"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Exception occurred during copying: {e}")

    return {td.Speaker_Name: td for td in speaker_data}
