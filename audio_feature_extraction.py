import warnings

import librosa.feature
import numpy as np
import os
import tqdm
import concurrent.futures

import AudioFile

from audio_balancer import TrainingData


def load_features(normv: float, use_config: bool = True, audio_data_dir: str = None, balanced: bool = False, normalized: bool = False, denoised: bool = False,
                  mfcc: bool = False, chroma: bool = False, spec_contrast: bool = False, split: bool = False, tonnetz: bool = False, n_mfcc: int = None,
                  selected: dict[str, TrainingData] = None) -> ([np.ndarray], [str]):
    data = []
    labels = []

    if use_config:
        import configuration
        config = configuration.read_config()

        audio_data_dir = config["Paths"]["feature data"]
        balanced = config.getboolean("Settings", "balance")
        normalized = config.getboolean("Settings", "normalize")
        split = config.getboolean("Settings", "split files")
        denoised = config.getboolean("Settings", "reduce noise")
        mfcc = config.getboolean("Features", "mfcc")
        chroma = config.getboolean("Features", "chroma")
        spec_contrast = config.getboolean("Features", "spec contrast")
        tonnetz = config.getboolean("Features", "tonnetz")
        n_mfcc = config.getint("Settings", "n mfcc")

        if not config.getboolean("Settings", "average amplitude"):
            normv = config.getfloat("Settings", "target amplitude")

    if not selected:
        # Iterate over all speakers (directories) in the root directory
        for speaker in os.listdir(audio_data_dir):
            files = []

            if speaker == "unused":
                continue

            speaker_dir: str = os.path.join(audio_data_dir, speaker)

            for file in os.listdir(os.path.join(audio_data_dir, speaker_dir)):
                if file.endswith(".npy"):
                    af = AudioFile.AudioFile(file, speaker)

                    if balanced == af.balanced and normalized == af.normalized and denoised == af.denoised and mfcc == af.mfcc and chroma == af.chroma and spec_contrast == af.speccontrast and tonnetz == af.tonnetz and split == af.split:
                        if mfcc and n_mfcc != af.mfcc_val:
                            continue
                        if normalized and (np.round(normv, 2) != af.norm_val):
                            continue

                        files.append(np.load(os.path.join(speaker_dir, file)))
                else:
                    warnings.warn(f"file {file} is not of type .npy and has been ignored")
                    continue
            data.extend(files)
            labels.extend([speaker] * len(files))

        # Convert data and labels to numpy arrays
        data = np.array(data)
        labels = np.array(labels)

        return data, labels
    else:
        for speaker in selected.keys():
            files = []

            speaker_dir: str = os.path.join(audio_data_dir, speaker)

            for file in selected.get(speaker).Speaker_Files:

                af = AudioFile.AudioFile(file[:-4] + ".npy", speaker)

                if mfcc:
                    af.mfcc = True
                    af.mfcc_val = n_mfcc

                af.chroma = chroma
                af.speccontrast = spec_contrast
                af.tonnetz = tonnetz

                data_path = os.path.join(audio_data_dir, af.speaker_name, af.generate_filename())

                if os.path.exists(data_path):
                    if balanced == af.balanced and normalized == af.normalized and denoised == af.denoised and mfcc == af.mfcc and chroma == af.chroma and spec_contrast == af.speccontrast and tonnetz == af.tonnetz and split == af.split:
                        if mfcc and n_mfcc != af.mfcc_val:
                            continue
                        if normalized and (np.round(normv, 2) != af.norm_val):
                            continue

                        files.append(np.load(data_path))
                else:
                    m = f"Data file for {file} was not found and it has been skipped"
                    warnings.warn(m)
                    continue
            data.extend(files)
            labels.extend([speaker] * len(files))

        # Convert data and labels to numpy arrays
        data = np.array(data)
        labels = np.array(labels)

        return data, labels


class RainbowColorGenerator:
    def __init__(self):
        self.rainbow_colors = [
            "#FF0000",  # Red
            "#FF7F00",  # Orange
            "#FFFF00",  # Yellow
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#4B0082",  # Indigo
            "#9400D3"  # Violet
        ]
        self.current_index = 0

    def next_color(self):
        color = self.rainbow_colors[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.rainbow_colors)
        return color


def extract_mfccs_features(audio: np.ndarray, sample_rate: float, n_mfcc: int = 40) -> np.ndarray:
    """
    MFCCs (Mel-frequency cepstral coefficients) provide a small set of features
    which concisely describe the overall shape of a spectral envelope.
    In MIR, it is often used to describe timbre.
    """
    # print(f"Extracting MFCCs features from {filename}")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


def extract_chroma_features(audio: np.ndarray, sample_rate: float, n_chroma: int = 24) -> np.ndarray:
    """
    Chroma features are an interesting and powerful representation for music audio
    in which the entire spectrum is projected onto 12 bins representing the 12
    distinct semitones (or chroma) of the musical octave.
    """
    # print(f"Extracting Chroma features from {filename}")
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_chroma=n_chroma)
    return np.mean(chroma.T, axis=0)


def extract_spec_contrast_features(audio: np.ndarray, sample_rate: float, n_bands: int = 6) -> np.ndarray:
    """
    Spectral contrast is defined as the difference in amplitude between peaks and valleys in a sound spectrum.
    It provides a measure of spectral shape that has been shown to be important in the perception of timbre.
    """
    # print(f"Extracting Spectral Contrast features from {filename}")
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=n_bands)
    return np.mean(spec_contrast.T, axis=0)


def extract_tonnetz_features(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    The tonnetz is a representation of musical pitch classes that has been used to analyze harmony in Western tonal music.
    It can be useful for key detection and chord recognition.
    """
    # print(f"Extracting Tonnetz features from {filename}")
    # chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=24, n_octaves=7)
    # tonnetz = librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    return np.mean(tonnetz.T, axis=0)


def extract_file_features(file_path: str, extract_mfcc: bool = False, extract_chroma: bool = False, extract_spec_contrast: bool = False,
                          extract_tonnetz: bool = False, n_mfcc: int = 40) -> np.ndarray:
    if not (extract_mfcc or extract_chroma or extract_spec_contrast or extract_tonnetz):
        raise ValueError("At least one feature set needs to be selected.")

    futures = []

    if not file_path.endswith(".wav"):
        raise ValueError("File needs to be wav to extract features.")

    audio, sample_rate = librosa.load(file_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks to executor

        if extract_mfcc:
            futures.append(executor.submit(extract_mfccs_features, audio, sample_rate, n_mfcc))
        if extract_chroma:
            futures.append(executor.submit(extract_chroma_features, audio, sample_rate))
        if extract_spec_contrast:
            futures.append(executor.submit(extract_spec_contrast_features, audio, sample_rate))
        if extract_tonnetz:
            futures.append(executor.submit(extract_tonnetz_features, audio, sample_rate))

    # Concatenate all features into one array ORDER MATTERS

    feature_sets_results = []

    for future in feature_sets_results:
        feature_sets_results.append(future.result())

    features = np.concatenate(feature_sets_results)

    return features


def extract_file_features_multi_threaded(file_path: str, data_dir: str, threads=4, overwrite: bool = False, extract_mfcc: bool = False,
                                         extract_chroma: bool = False, extract_spec_contrast: bool = False, extract_tonnetz: bool = False,
                                         n_mfcc: int = 40, ) -> None:
    af = AudioFile.AudioFile(os.path.basename(file_path[:-4]) + ".npy", os.path.dirname(file_path).split('/')[-1])

    if extract_mfcc:
        af.mfcc = True
        af.mfcc_val = n_mfcc

    af.chroma = extract_chroma
    af.speccontrast = extract_spec_contrast
    af.tonnetz = extract_tonnetz

    data_path = os.path.join(data_dir, af.generate_filename())

    if os.path.exists(data_path) and not overwrite:
        return

    else:

        # Load the audio file
        audio, sample_rate = librosa.load(file_path)

        feature_sets = []

        # Use a ThreadPoolExecutor to execute the functions in separate threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks to executor

            if extract_mfcc:
                mfccs = executor.submit(extract_mfccs_features, audio, sample_rate, n_mfcc)
                feature_sets.append(mfccs)
            if extract_chroma:
                chrom = executor.submit(extract_chroma_features, audio, sample_rate)
                feature_sets.append(chrom)
            if extract_spec_contrast:
                spec_contrast = executor.submit(extract_spec_contrast_features, audio, sample_rate)
                feature_sets.append(spec_contrast)
            if extract_tonnetz:
                tonnetz = executor.submit(extract_tonnetz_features, audio, sample_rate)
                feature_sets.append(tonnetz)

        # Concatenate all features into one array ORDER MATTERS
        feature_sets_results = []

        for future in feature_sets:
            feature_sets_results.append(future.result())

        features = np.concatenate(feature_sets_results)

        np.save(data_path, features)


def extract_features_multi_threaded(use_config: bool = True, input_dir: str = None, data_dir: str = None, threads: int = 4, over_write: bool = False,
                                    extract_mfcc: bool = False, nmfcc: int = 30, extract_chroma: bool = False, extract_spec_contrast: bool = False,
                                    extract_tonnetz: bool = False, selected: dict[str, TrainingData] = None) -> None:
    if use_config:
        import configuration
        config = configuration.read_config()
        input_dir = config["Paths"]["training data"]
        data_dir = config["Paths"]["feature data"]
        over_write = config.getboolean("Settings", "overwrite data")

        extract_mfcc = config.getboolean("Features", "mfcc")
        nmfcc = config.getint("Settings", "N MFCC")
        extract_chroma = config.getboolean("Features", "chroma")
        extract_spec_contrast = config.getboolean("Features", "spec contrast")
        extract_tonnetz = config.getboolean("Features", "tonnetz")

    futures = []
    if not selected:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker in os.listdir(input_dir):

                if speaker == "unused":
                    continue

                os.makedirs(os.path.join(data_dir, speaker), exist_ok=True)

                speaker_files = os.listdir(os.path.join(input_dir, speaker))
                for file in speaker_files:
                    futures.append(
                        executor.submit(
                            extract_file_features_multi_threaded,
                            os.path.join(input_dir, speaker, file),
                            os.path.join(data_dir, speaker),
                            threads,
                            over_write,
                            extract_mfcc,
                            extract_chroma,
                            extract_spec_contrast,
                            extract_tonnetz,
                            nmfcc,
                        )
                    )

            for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Extracting features",
                    dynamic_ncols=True,
                    colour="blue",
            ):
                try:
                    future.result()
                except Exception as e:
                    print(e)

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker in selected.keys():

                os.makedirs(os.path.join(data_dir, speaker), exist_ok=True)

                for file in selected.get(speaker).Speaker_Files:
                    futures.append(
                        executor.submit(
                            extract_file_features_multi_threaded,
                            os.path.join(input_dir, speaker, file),
                            os.path.join(data_dir, speaker),
                            threads,
                            over_write,
                            extract_mfcc,
                            extract_chroma,
                            extract_spec_contrast,
                            extract_tonnetz,
                            nmfcc,
                        )
                    )

            for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Extracting features",
                    dynamic_ncols=True,
                    colour="#FF7F00",
            ):
                try:
                    future.result()
                except Exception as e:
                    print(e)


def extract_all_features_multi_threaded(use_config: bool = True, input_dir: str = None, data_dir: str = None, threads: int = 4, nmfcc: int = 20,
                                        over_write: bool = False,
                                        selected: dict[str, TrainingData] = None) -> None:
    if use_config:
        import configuration
        config = configuration.read_config()
        input_dir = config["Paths"]["training data"]
        data_dir = config["Paths"]["feature data"]
        over_write = config.getboolean("Settings", "overwrite data")

        extract_mfcc = config.getboolean("Features", "mfcc")
        nmfcc = config.getint("Settings", "N MFCC")
        extract_chroma = config.getboolean("Features", "chroma")
        extract_spec_contrast = config.getboolean("Features", "spec contrast")
        extract_tonnetz = config.getboolean("Features", "tonnetz")

    futures = []

    speakers = []
    if not selected:
        speakers = os.listdir(input_dir)
    else:
        speakers = selected.keys()
        rgb = RainbowColorGenerator()

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker in speakers:
                if speaker == "unused":
                    continue

                os.makedirs(os.path.join(data_dir, speaker), exist_ok=True)

                if not selected:
                    speaker_files = os.listdir(os.path.join(input_dir, speaker))
                else:
                    speaker_files = selected.get(speaker).Speaker_Files
                for file in speaker_files:
                    futures.append(
                        executor.submit(
                            extract_file_features_multi_threaded,
                            os.path.join(input_dir, speaker, file),
                            os.path.join(data_dir, speaker),
                            threads,
                            over_write,
                            True,
                            False,
                            False,
                            False,
                            nmfcc,
                        )
                    )

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Extracting MFCC features", dynamic_ncols=True,
                                    colour=rgb.next_color(), ):
                try:
                    future.result()
                except Exception as e:
                    print(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker in speakers:
                if speaker == "unused":
                    continue

                os.makedirs(os.path.join(data_dir, speaker), exist_ok=True)

                if not selected:
                    speaker_files = os.listdir(os.path.join(input_dir, speaker))
                else:
                    speaker_files = selected.get(speaker).Speaker_Files
                for file in speaker_files:
                    futures.append(
                        executor.submit(
                            extract_file_features_multi_threaded,
                            os.path.join(input_dir, speaker, file),
                            os.path.join(data_dir, speaker),
                            threads,
                            over_write,
                            False,
                            True,
                            False,
                            False,
                            40,
                        )
                    )

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Extracting Chroma features", dynamic_ncols=True,
                                    colour=rgb.next_color(), ):
                try:
                    future.result()
                except Exception as e:
                    print(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker in speakers:
                if speaker == "unused":
                    continue

                os.makedirs(os.path.join(data_dir, speaker), exist_ok=True)

                if not selected:
                    speaker_files = os.listdir(os.path.join(input_dir, speaker))
                else:
                    speaker_files = selected.get(speaker).Speaker_Files
                for file in speaker_files:
                    futures.append(
                        executor.submit(
                            extract_file_features_multi_threaded,
                            os.path.join(input_dir, speaker, file),
                            os.path.join(data_dir, speaker),
                            threads,
                            over_write,
                            False,
                            False,
                            True,
                            False,
                            40,
                        )
                    )

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Extracting Spectral Contrast features",
                                    dynamic_ncols=True,
                                    colour=rgb.next_color(), ):
                try:
                    future.result()
                except Exception as e:
                    print(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for speaker in speakers:
                if speaker == "unused":
                    continue

                os.makedirs(os.path.join(data_dir, speaker), exist_ok=True)

                if not selected:
                    speaker_files = os.listdir(os.path.join(input_dir, speaker))
                else:
                    speaker_files = selected.get(speaker).Speaker_Files
                for file in speaker_files:
                    futures.append(
                        executor.submit(
                            extract_file_features_multi_threaded,
                            os.path.join(input_dir, speaker, file),
                            os.path.join(data_dir, speaker),
                            threads,
                            over_write,
                            False,
                            False,
                            False,
                            True,
                            40,
                        )
                    )

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Extracting Tonnetz features", dynamic_ncols=True,
                                    colour=rgb.next_color(), ):
                try:
                    future.result()
                except Exception as e:
                    print(e)
