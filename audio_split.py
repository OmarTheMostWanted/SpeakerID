import os
import librosa
import concurrent.futures
from tqdm import tqdm
import soundfile as sf


def split_audio_librosa(input_path: str, output_path: str = None):
    # Load the audio file using librosa
    data, rate = librosa.load(input_path, sr=None)

    # Convert to milliseconds and discard the first and last minute
    data = data[rate * 60:-rate * 60]

    # Split audio into 10-minute chunks
    for i in range(0, len(data), rate * 600):
        chunk = data[i:i + rate * 600]
        # Discard if chunk is less than 10 minutes
        if len(chunk) == rate * 600:
            if output_path is not None:
                # Save the result using soundfile
                sf.write(output_path + f"_{i // (rate * 600)}.wav", chunk, rate)


def split_audio_multi_thread(input_dir: str, output_dir: str, threads: int = 4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for speaker_dir in os.listdir(input_dir):

            if not os.path.exists(os.path.join(output_dir , speaker_dir)):
                os.makedirs(os.path.join(output_dir , speaker_dir))

            for file in os.listdir(os.path.join(input_dir, speaker_dir)):
                if file.endswith(".wav"):
                    audio_path = os.path.join(os.path.join(input_dir, speaker_dir), file)
                    output_path = os.path.join(os.path.join(output_dir, speaker_dir), file[:-4] + "_split")

                    futures.append(executor.submit(split_audio_librosa, audio_path, output_path))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), dynamic_ncols=True,
                           desc="Splitting audio files", colour="#4B0082"):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred during splitting: {e}")


split_audio_multi_thread("/home/tmw/Digivox/audio_data/data_normalized", "/home/tmw/Digivox/audio_data/data_split")
