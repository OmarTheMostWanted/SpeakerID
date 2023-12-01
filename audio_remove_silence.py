from pydub import AudioSegment
from pydub.silence import split_on_silence
import concurrent.futures
from tqdm import tqdm
import os


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

                    futures.append(executor.submit(remove_silence_from_audio, audio_path, desilenced_path))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                           desc="Removing silence", colour="#9400D3"):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred during removing silence: {e}")
