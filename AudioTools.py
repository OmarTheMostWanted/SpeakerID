import os
import concurrent.futures
from pydub import AudioSegment
from tqdm import tqdm
import sys
import soundfile as sf


def convert_file_to_wav(audio_path, wav_path, audio_format):
    # Convert audio to wav
    if audio_format == 'mp3':
        audio = AudioSegment.from_mp3(audio_path)
    elif audio_format == 'm4a':
        audio = AudioSegment.from_file(audio_path, format='m4a')
    else:
        print("Unsupported format")
        return
    audio.export(wav_path, format='wav')


def convert_to_wav(input_dir, output_dir, threads=10):
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


def get_audio_files_duration(audio_dir: str) -> float:
    """Calculate the total duration of audio files in seconds."""
    total_duration = 0.0
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            audio_file = AudioSegment.from_wav(os.path.join(audio_dir, file))
            total_duration += audio_file.duration_seconds
    return total_duration


def select_audio_files(speaker_dirs: [str], target_duration: float) -> [str]:
    """Select audio files for each speaker to match the target duration."""
    speaker_files = {}
    for speaker_dir in speaker_dirs:
        print(f"Processing {speaker_dir}...")
        audio_files = [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.endswith('.wav')]
        audio_files.sort(key=lambda f: -os.path.getsize(f))  # Sort by file size in descending order
        selected_files = []
        selected_duration = 0.0
        for audio_file in audio_files:
            file_duration = get_audio_files_duration([audio_file])
            if selected_duration + file_duration <= target_duration:
                selected_files.append(audio_file)
                selected_duration += file_duration
        speaker_files[speaker_dir] = selected_files
        print(f"Selected {len(selected_files)} files with total duration {selected_duration} seconds.")
    return speaker_files


def main():
    root_dir = "audio_files_wav"  # Replace with actual path
    speaker_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    duration = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for speaker_dir in speaker_dirs:
            futures.append(executor.submit(get_audio_files_duration, os.listdir(speaker_dir)))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ncols=70):
            try:
                duration.append(future.result())
                print(f"Duration for {speaker_dir}: {duration} seconds.")
            except Exception as e:
                print(f"Exception occurred during duration finding: {e}")

    min_duration = min(duration)
    print(f"Target duration for each speaker: {min_duration} seconds.")
    speaker_files = select_audio_files(speaker_dirs, min_duration)
    print("Selection completed.")
    return speaker_files


def main2():
    root_dir = "audio_files_wav"  # Replace with actual path
    speaker_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    duration = []

    total_clips = 0

    for speaker_dir in speaker_dirs:
        total_clips += len(os.listdir(speaker_dir))

    for speaker_dir in speaker_dirs:
        duration.append(get_audio_files_duration(os.listdir(speaker_dir)))

    min_duration = min(duration)
    print(f"Target duration for each speaker: {min_duration} seconds.")
    speaker_files = select_audio_files(speaker_dirs, min_duration)
    print("Selection completed.")
    return speaker_files


def get_the_speaker_with_smallest_dataset(speakers_dir: str) -> (str, float):
    arr = []
    min = float("inf")
    speaker_min: str
    print(f"Getting the maximum number of audio seconds to use per speaker")
    for speaker in tqdm(os.listdir("audio_files_wav"), total=len(os.listdir("audio_files_wav")), colour="blue",
                        dynamic_ncols=True):
        duration = get_audio_files_duration(os.path.join("audio_files_wav/", speaker))
        arr.append(duration)
        if duration < min:
            min = duration
            speaker_min = speaker
    print(f"The speaker with smallest data set is {speaker_min} with duration of {min}")
    return speaker_min, min


if __name__ == '__main__':
    if len(sys.argv) == 1:
        get_the_speaker_with_smallest_dataset("audio_files_wav")

    if len(sys.argv) == 4 and sys.argv[1] == 'convert':
        convert_to_wav(sys.argv[2], sys.argv[3])
    else:
        print('Usage: AudioTools.py convert <input_dir> <output_dir>')
        print('Example: AudioTools.py convert ./input ./output')
