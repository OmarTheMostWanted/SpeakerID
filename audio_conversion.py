import os
import concurrent.futures
from pydub import AudioSegment
from tqdm import tqdm
import sys


def normalize_audio_file(input_path, output_path, target_amplitude=-20.0):
    audio = AudioSegment.from_file(input_path)

    # Calculate the difference in dB between the target amplitude and the current amplitude
    diff = target_amplitude - audio.dBFS

    # Normalize the audio to the target amplitude
    normalized_audio = audio + diff

    # Export the normalized audio to the output path
    normalized_audio.export(output_path, format="wav")


def remove_noise_file(input_audio_path, output_audio_path, noise_profile_path):
    audio = AudioSegment.from_file(input_audio_path)
    noise = AudioSegment.from_file(noise_profile_path)

    # Remove noise using the noise profile
    cleaned_audio = audio.overlay(noise, position=0)

    # Export the cleaned audio to the output path
    cleaned_audio.export(output_audio_path, format="wav")


def convert_file_to_wav(audio_path, wav_path, audio_format):
    # Convert audio to wav
    if audio_format == 'mp3':
        audio = AudioSegment.from_mp3(audio_path)
    elif audio_format == 'm4a':
        audio = AudioSegment.from_file(audio_path, format='m4a')
    audio.export(wav_path, format='wav')


def convert_to_wav(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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


# def remove_noise(input_dir, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     futures = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#         for root, dirs, files in os.walk(input_dir):
#             for file in files:
#                 if file.endswith('.wav'):
#                     wav_path = os.path.join(root, file)
#                     rel_root = os.path.relpath(root, input_dir)
#                     denoised_path = os.path.join(output_dir, rel_root, file[:-4] + '_denoised.wav')

#                     # Create new directories in output_dir as necessary
#                     os.makedirs(os.path.dirname(wav_path), exist_ok=True)

#                     # Submit a new task to the thread pool
#                     futures.append(executor.submit(remove_noise_file, wav_path, denoised_path))

#         # Add progress bar using tqdm
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ncols=70):
#             try:
#                 future.result()
#             except Exception as e:
#                 print(f"Exception occurred during conversion: {e}")

def normalize_audio(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    rel_root = os.path.relpath(root, input_dir)
                    normalized_path = os.path.join(output_dir, rel_root, file[:-4] + '_normalized.wav')

                    # Create new directories in output_dir as necessary
                    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

                    # Submit a new task to the thread pool
                    futures.append(executor.submit(normalize_audio_file, wav_path, normalized_path))

        # Add progress bar using tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), ncols=70):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred during conversion: {e}")


# Usage:
# remove_noise_file("/home/tmw/Code/ML/audio_files_wav", '/home/tmw/Code/ML/audio_files_denoised')

# normalize_audio_file("/home/tmw/Code/ML/audio_files_wav/", '/home/tmw/Code/ML/audio_files_normalized/')

if __name__ == "__main__":

    if len(sys.argv) == 3:
        convert_to_wav(sys.argv[1], sys.argv[2])
