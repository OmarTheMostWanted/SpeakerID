import warnings

from pydub import AudioSegment
from pydub.silence import split_on_silence
import concurrent.futures
from tqdm import tqdm
import os

import numpy as np
import soundfile as sf
import librosa

import numpy as np


def is_silent(audio_file, threshold=80):
    audio_data, _ = librosa.load(audio_file)
    return audio_data.max() < threshold


# Define the function to remove silence from an audio file
def remove_silence_from_audio_librosa(file_path, output, sr=22050, frame_length=1024, hop_length=512):
    if os.path.exists(output):
        return

    if is_silent(file_path):
        return

        # Load the audio file using librosa's load function
    # 'sr' is the target sampling rate (set to 22050 by default)
    # 'y' is the audio time series and 'sr' is the sampling rate
    y, sr = librosa.load(file_path, sr=None)

    # Trim the silence from the audio using librosa's trim function
    # 'frame_length' is the length of the analysis window (set to 1024 by default)
    # 'hop_length' is the number of samples between successive frames (set to 512 by default)
    # 'top_db' is the threshold (in decibels) below which audio is considered silent (set to 20 by default)
    # 'y_trim' is the trimmed audio signal and 'index' are the start and end indices of the non-silent intervals
    y_trim, index = librosa.effects.trim(y, frame_length=frame_length, hop_length=hop_length, top_db=80)

    if len(y_trim) < frame_length:
        warnings.warn(f"Warning: output audio is very short. The top_db threshold might be too high.")
        return

    # Write the trimmed audio signal back to a new file using soundfile's write function
    # 'output.wav' is the name of the output file (replace with your desired output file path)
    # 'y_trim' is the audio data and 'sr' is the sampling rate
    sf.write(output, y_trim, sr)


def remove_silence_from_audio(file_path, output):
    # Load your audio file
    audio = AudioSegment.from_file(file_path)

    # Split track where the silence is 2 seconds or more and get chunks using
    # the imported function.
    chunks = split_on_silence(
        # Use the loaded audio.
        audio,
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len=2000,
        # Consider a chunk silent if it's quieter than -50 dBFS.
        silence_thresh=-50
    )

    # Initialize final audio
    final_audio = chunks[0]

    # Go through each chunk and append the chunk to final audio
    for chunk in chunks[1:]:
        final_audio += chunk

    # Export the audio to the output path
    final_audio.export(output, format="wav")

    return final_audio


def remove_silence_multi_thread(threads: int = 4, use_conf: bool = True, input_dir: str = None, out_put_dir: str = None, ):
    if out_put_dir is not None and not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    if use_conf:
        import configuration
        config = configuration.read_config()
        if not config.getboolean("Settings", "Remove Silence"):
            print("Silence removal is disabled in the configuration file, so this step has been skipped")
            return
        if config.getboolean("Settings", "Reduce Noise"):
            input_dir = config["Paths"]["denoised files"]
        elif config.getboolean("Settings", "Convert to wav"):
            input_dir = config["Paths"]["wav files"]
        else:
            input_dir = config["Paths"]["raw files"]

        out_put_dir = config["Paths"]["remove silence files"]

    elif input_dir is None or out_put_dir is None:
        raise Exception("Provide a directory or use config")

    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for speaker_dir in os.listdir(input_dir):
            os.makedirs(os.path.join(out_put_dir, speaker_dir), exist_ok=True)
            for file in os.listdir(os.path.join(input_dir, speaker_dir)):
                if file.endswith(".wav"):
                    audio_path = os.path.join(input_dir, speaker_dir, file)
                    desilenced_path = os.path.join(out_put_dir, speaker_dir, file[:-4] + "_desilenced.wav")

                    futures.append(executor.submit(remove_silence_from_audio_librosa, audio_path, desilenced_path))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                           desc="Removing silence", colour="#9400D3"):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred during removing silence: {e}")


if __name__ == "__main__":
    remove_silence_from_audio_librosa("/home/tmw/Digivox/audio_data/Test/00242111_000.WAV", "/home/tmw/Digivox/audio_data/Test/00242111_000_rs.WAV")
    os.remove("/home/tmw/Digivox/audio_data/Test/00242111_000_rs.WAV")
# Here are a couple of methods for automatically removing silent parts from audio recordings locally:
#
# **Method 1: Using Audacity**
#
# Audacity is a free and open-source audio editing software that can be used to remove silent parts from audio recordings. Here's how to do it:
#
# 1. Import your audio recording into Audacity.
# 2. Select the "Silence" tool from the toolbar.
# 3. Click and drag the mouse over the silent parts of the recording to select them.
# 4. Press the "Delete" key to remove the selected silent parts.
# 5. Repeat steps 3 and 4 until all of the silent parts have been removed.
# 6. Export the edited audio recording.
#
# **Method 2: Using ffmpeg**
#
# ffmpeg is a command-line tool that can be used to manipulate audio and video files. Here's how to remove silent parts from audio recordings using ffmpeg:
#
# 1. Open a terminal window.
# 2. Navigate to the directory containing your audio recording.
# 3. Use the following command to remove silent parts from the audio recording:
#
#
# ffmpeg -i input.wav -af silenceremove=start_duration=-1:end_duration=-1:duration=1:detection_threshold=-35dB output.wav
#
#
# This command will remove all silent parts from the audio recording that are longer than one second. The '-35dB' parameter specifies the minimum decibel level that must be exceeded for audio to be considered non-silent.
#
# **Additional Notes**
#
# * The threshold value used to detect silence can be adjusted to remove shorter or longer silent parts.
# * You can also use these methods to remove gaps between speech segments in a podcast or to trim the beginning and end of an audio recording.
