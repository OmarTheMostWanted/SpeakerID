from pydub import AudioSegment
from pydub.silence import split_on_silence
import concurrent.futures
from tqdm import tqdm
import os

import numpy as np
import soundfile as sf
import librosa


import numpy as np

def remove_silence_from_audio_librosa(file_path, output, frame_length=1024, hop_length=512):
    # Load the audio file at its native sample rate
    y, sr = librosa.load(file_path, sr=22850)

    # Estimate the decibel level of the background noise
    # Here, we take the mean of the absolute amplitudes of the first 2 seconds of the audio
    # You might want to adjust this depending on the characteristics of your audio files
    noise_amp = np.mean(np.abs(y[:sr*2]))  # sr is the number of samples per second
    top_db = 20 * np.log10(noise_amp) + 2  # add 2 dB

    top_db = -50

    # Trim the silence from the audio
    y_trim, index = librosa.effects.trim(y, frame_length=frame_length, hop_length=hop_length, top_db=top_db)

    # Check if the resulting audio is too short
    if len(y_trim) < frame_length:
        print(f"Warning: output audio is very short: {len(y_trim)}, The top_db threshold might be too high.")
        return None, None

    # Write the trimmed audio signal back to a new file
    sf.write(output, y_trim, sr)

    # Return the trimmed audio signal and the sampling rate
    return y_trim, sr



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
    remove_silence_from_audio_librosa("/home/tmw/Digivox/Test/00210632_000_denoised.wav", "/home/tmw/Digivox/Test/00210632_000_denoised_rs.wav")


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

