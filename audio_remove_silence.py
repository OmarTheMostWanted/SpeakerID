from pydub import AudioSegment
from pydub.silence import split_on_silence
import concurrent.futures
from tqdm import tqdm
import os

# Import necessary libraries
import numpy as np
import soundfile as sf
import librosa


# Define the function to remove silence from an audio file
def remove_silence_from_audio_librosa(file_path, output, sr=22050, frame_length=1024, hop_length=512, top_db=-50):
    # Load the audio file using librosa's load function
    # 'sr' is the target sampling rate (set to 22050 by default)
    # 'y' is the audio time series and 'sr' is the sampling rate
    y, sr = librosa.load(file_path, sr=sr)

    # Trim the silence from the audio using librosa's trim function
    # 'frame_length' is the length of the analysis window (set to 1024 by default)
    # 'hop_length' is the number of samples between successive frames (set to 512 by default)
    # 'top_db' is the threshold (in decibels) below which audio is considered silent (set to 20 by default)
    # 'y_trim' is the trimmed audio signal and 'index' are the start and end indices of the non-silent intervals
    y_trim, index = librosa.effects.trim(y, frame_length=frame_length, hop_length=hop_length, top_db=top_db)

    # Write the trimmed audio signal back to a new file using soundfile's write function
    # 'output.wav' is the name of the output file (replace with your desired output file path)
    # 'y_trim' is the audio data and 'sr' is the sampling rate
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
        if config.getboolean("Settings", "Convert to wav"):
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
