import multiprocessing as mp
import librosa.feature
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import tqdm  # for the progress bar
import pickle
import concurrent.futures


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


def extract_chroma_features(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Chroma features are an interesting and powerful representation for music audio
    in which the entire spectrum is projected onto 12 bins representing the 12
    distinct semitones (or chroma) of the musical octave.
    """
    # print(f"Extracting Chroma features from {filename}")
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    return np.mean(chroma.T, axis=0)


def extract_spec_contrast_features(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Spectral contrast is defined as the difference in amplitude between peaks and valleys in a sound spectrum.
    It provides a measure of spectral shape that has been shown to be important in the perception of timbre.
    """
    # print(f"Extracting Spectral Contrast features from {filename}")
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    return np.mean(spec_contrast.T, axis=0)


def extract_tonnetz_features(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    The tonnetz is a representation of musical pitch classes that has been used to analyze harmony in Western tonal music.
    It can be useful for key detection and chord recognition.
    """
    # print(f"Extracting Tonnetz features from {filename}")
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    return np.mean(tonnetz.T, axis=0)


def extract_file_features(
        file_path: str,
        extract_mfcc: bool = False,
        extract_chroma: bool = False,
        extract_spec_contrast: bool = False,
        extract_tonnetz: bool = False,
        n_mfcc: int = 40,
) -> np.ndarray:
    if not (extract_mfcc or extract_chroma or extract_spec_contrast or extract_tonnetz):
        raise ValueError("At least one feature set needs to be selected.")

    futures = []

    if not file_path.endswith(".wav"):
        raise ValueError("File needs to be wav to extract features.")

    audio, sample_rate = librosa.load(file_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks to executor

        if extract_mfcc:
            futures.append(
                executor.submit(extract_mfccs_features, audio, sample_rate, n_mfcc)
            )
        if extract_chroma:
            futures.append(executor.submit(extract_chroma_features, audio, sample_rate))
        if extract_spec_contrast:
            futures(executor.submit(extract_spec_contrast_features, audio, sample_rate))
        if extract_tonnetz:
            futures.append(
                executor.submit(extract_tonnetz_features, audio, sample_rate)
            )

    # Concatenate all features into one array ORDER MATTERS

    feature_sets_results = []

    for future in feature_sets_results:
        feature_sets_results.append(future.result())

    features = np.concatenate(feature_sets_results)

    return features


def extract_file_features_multi_threaded(
        file_path: str,
        cache_dir: str,
        threads=4,
        overwrite: bool = False,
        extract_mfcc: bool = False,
        extract_chroma: bool = False,
        extract_spec_contrast: bool = False,
        extract_tonnetz: bool = False,
        n_mfcc: int = 40,
        use_cache: bool = True,
) -> None:
    cache_file_name = os.path.basename(file_path)[:-4]

    if extract_mfcc:
        cache_file_name += f"_mfcc({n_mfcc})"
    if extract_chroma:
        cache_file_name += "_chroma"
    if extract_spec_contrast:
        cache_file_name += "_speccontrast"
    if extract_tonnetz:
        cache_file_name += "_tonnetz"

    cache_file_name += ".npy"

    if (
            use_cache
            and os.path.exists(os.path.join(cache_dir, cache_file_name))
            and not overwrite
    ):
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

        if use_cache:
            np.save(os.path.join(cache_dir, cache_file_name), features)


class AudioFeatureFile:
    def __init__(self, speaker_name, file_name, features):
        self.speaker_name = speaker_name
        self.file_name = file_name
        self.features = features

    def get_file_name(self):
        feature_str = "_".join(self.features)
        return f"{self.speaker_name}/{self.file_name}_{feature_str}.npy"

    def parse_file_name(self, file_path):
        # Assuming the file path is in the correct format
        parts = os.path.splitext(file_path)[0].split('/')
        self.speaker_name = parts[0]
        self.file_name = parts[1].split('_')[0]
        self.features = [part for part in parts[1].split('_') if part not in ['mfcc', 'chroma', 'speccontrast', 'tonnetz']]

    @staticmethod
    def read_files_for_features(speaker_directory: str, features: [str], mfcc_number: int = None):
        files = []

        if "mfcc" in features:
            if mfcc_number is None:
                raise ValueError("mfcc_number must be provided if mfcc is selected")
            else:
                for file_name in os.listdir(speaker_directory):
                    if file_name.endswith(".npy"):
                        audio_file = AudioFeatureFile("", "", [])
                        audio_file.parse_file_name(os.path.join(speaker_directory, file_name))
                        if audio_file.features == features and f"mfcc({mfcc_number})" in file_name:
                            files.append(file_name)
                return files

        else:
            if mfcc_number is not None:
                raise ValueError("mfcc_number must be None if mfcc is not selected")
            else:
                for file_name in os.listdir(speaker_directory):
                    if file_name.endswith(".npy"):
                        audio_file = AudioFeatureFile("", "", [])
                        audio_file.parse_file_name(os.path.join(speaker_directory, file_name))
                return files


def load_features(audio_files_cache_dir: str, features: [str], n_mfcc: int = None) -> ([np.ndarray], [str]):
    data = []
    labels = []

    # Iterate over all speakers (directories) in the root directory
    for speaker in os.listdir(audio_files_cache_dir):
        files: [str]

        speaker_dir = os.path.join(audio_files_cache_dir, speaker)

        files = AudioFeatureFile.read_files_for_features(speaker_dir, features, n_mfcc)

        data.extend(np.load(os.path.join(speaker_dir, files)))

    labels.extend([speaker] * len(files))

    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data, labels


def extract_features_multi_threaded(
        audio_files_dir: str,
        cache_dir: str,
        threads: int = 4,
        over_write: bool = False,
        extract_mfcc: bool = False,
        extract_chroma: bool = False,
        extract_spec_contrast: bool = False,
        extract_tonnetz: bool = False,
) -> None:
    color = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for speaker in os.listdir(audio_files_dir):
            os.makedirs(os.path.join(cache_dir, speaker), exist_ok=True)

            speaker_files = os.listdir(os.path.join(audio_files_dir, speaker))
            for file in speaker_files:
                futures.append(
                    executor.submit(
                        extract_file_features_multi_threaded,
                        os.path.join(audio_files_dir, speaker, file),
                        os.path.join(cache_dir, speaker),
                        4,
                        over_write,
                        extract_mfcc,
                        extract_chroma,
                        extract_spec_contrast,
                        extract_tonnetz,
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


def TrainSupportVectorClassification(data, labels):
    # Encode labels to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=None
    )

    # Define the model
    model = SVC(probability=True, verbose=False)

    # Train the model and add a progress message
    print("Training model...")
    model.fit(X_train, y_train)

    # Test the model and add a progress message
    print("Testing model...")
    y_pred = model.predict(X_test)

    # Calculate the accuracy of our model and print it out with more information
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return (model, le, accuracy)


def save_model(dist, model, le, accuracy, features: [str]):
    # Save the trained model and add a progress message
    print("Saving model...")
    with open(
            os.path.join(dist, f"model_{np.round(accuracy * 100, 2)}{features}_.pkl", "wb")
    ) as f:
        pickle.dump(model, f)

    # Save the label encoder and add a progress message
    print("Saving label encoder...")
    with open(
            os.path.join(
                dist, f"label_encoder_{np.round(accuracy * 100, 2)}{features}_.pkl", "wb"
            )
    ) as f:
        pickle.dump(le, f)


def load_model(dir, model_file, label_encoder_file):
    # Load the trained model and add a progress message
    print("Loading model...")
    with open(os.path.join(dir, model_file), "rb") as f:
        model = pickle.load(f)

    # Load the label encoder and add a progress message
    print("Loading label encoder...")
    with open(os.path.join(dir, label_encoder_file), "rb") as f:
        le = pickle.load(f)

    # Now you can use `model.predict_proba()` to get prediction probabilities for new data
    return model, le


# def predict_speaker(model, le, threads=4):
#     while True:
#         # Ask user for an audio file path and identify it.
#         audio_file_path = input("Please enter an audio file path or x to close: ")
#         if audio_file_path == "x":
#             quit(0)
#         features = threads > 0 and extract_file_features_multi_threaded(audio_file_path, threads) or extract_file_features(
#             audio_file_path)
#         speaker_id = model.predict([features])
#         speaker_name = le.inverse_transform(speaker_id)
#         print(f"The speaker is: {speaker_name[0]}")
#         return speaker_name
#
#


def predict_speaker_with_probability(model, le, threads=0):
    while True:
        # Ask user for an audio file path and identify it.
        audio_file_path = input("Please enter an audio file path or 'x' to close: ")
        if audio_file_path.lower() == "x":
            print("Closing the program.")
            break

        if not audio_file_path.endswith(".wav"):
            print("only wav files are supported")
            continue

        import audio_tools as at

        print("normalizing file")
        at.normalize_file(audio_file_path, "temp_normalized.wav", target_amplitude=-26.6)

        print("applying noise reduction file")
        at.reduce_noise("temp_normalized.wav", output_path="temp_normalized_denoised.wav")

        audio, sample_rate = librosa.load("temp_normalized_denoised.wav")

        features = extract_spec_contrast_features(audio, sample_rate)

        speaker_id = model.predict([features])
        speaker_name = le.inverse_transform(speaker_id)

        # Get the probability of the prediction
        probability = model.predict_proba([features])
        max_prob_index = np.argmax(probability)
        max_prob = probability[0][max_prob_index]

        print(
            f"The speaker is: {speaker_name[0]} with a probability of {max_prob * 100:.2f}%"
        )


def save_data(data, labels):
    # Save the data and labels for future use
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)

    with open("labels.pkl", "wb") as f:
        pickle.dump(labels, f)


def load_data():
    data = []
    labels = []

    # Load the data and labels
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    with open("labels.pkl", "rb") as f:
        labels = pickle.load(f)

    return data, labels


def train(features: [str], threads=4) -> str:
    import configuration
    import audio_tools as at

    conf = configuration.read_config()

    models_dir = conf["Paths"]["models"]

    at.balance_audio_multi_thread(use_conf=True, threads=15)
    at.normalize_audio_files_multi_thread(use_conf=True, threads=15)
    at.reduce_noise_multi_thread(threads=15, use_conf=True)

    mfcc = "mfcc" in features
    chroma = "chroma" in features
    spec_contrast = "contrast" in features
    tonnetz = "tonnetz" in features

    extract_features_multi_threaded(
        audio_files_dir=conf["Paths"]["training data"],
        cache_dir=conf["Paths"]["data cache"],
        threads=4,
        extract_mfcc=mfcc,
        extract_chroma=chroma,
        extract_spec_contrast=spec_contrast,
        extract_tonnetz=tonnetz,
        over_write=False,
    )

    data, labels = load_features(conf["Paths"]["data cache"])

    model, le, accuracy = TrainSupportVectorClassification(data, labels)

    save_model(models_dir, model, le, accuracy, features)

    return model, le, accuracy


if __name__ == "__main__":
    # Example usage:
    # Creating an instance of AudioFeatureFile
    audio_file_instance = AudioFeatureFile("John", "song", ["mfcc", "chroma"])

    # Generating a file name
    file_name = audio_file_instance.get_file_name()
    print(f"Generated file name: {file_name}")

    # Parsing a file name
    parsed_audio_file = AudioFeatureFile("", "", [])
    parsed_audio_file.parse_file_name(file_name)
    print(f"Parsed speaker name: {parsed_audio_file.speaker_name}")
    print(f"Parsed file name: {parsed_audio_file.file_name}")
    print(f"Parsed features: {parsed_audio_file.features}")

    # Reading files for a given set of features and mfcc number
    speaker_directory = "John"
    features_to_find = ["mfcc", "speccontrast"]
    mfcc_number_to_find = 30
    matching_files = AudioFeatureFile.read_files_for_features(speaker_directory, features_to_find, mfcc_number_to_find)
    print(f"Matching files: {matching_files}")

    model, le = load_model(
        "/home/tmw/Code/SpeakerID", "model_100.0_.pkl", "label_encoder_100.0_.pkl"
    )

    predict_speaker_with_probability(model, le)
