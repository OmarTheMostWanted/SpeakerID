import multiprocessing as mp
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import tqdm  # for the progress bar
import pickle
import concurrent.futures


def extract_file_features(filename: str) -> [np.ndarray]:
    # Load the audio file
    audio, sample_rate = librosa.load(filename)

    # Extract MFCCs from the audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)

    # Extract Spectral Contrast features
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)

    # Extract Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    tonnetz_processed = np.mean(tonnetz.T, axis=0)

    # Concatenate all features into one array
    features = np.concatenate(
        [mfccs_processed, chroma_processed, spec_contrast_processed, tonnetz_processed]
    )
    return features


def extract_mfccs(audio: np.ndarray, sample_rate: int, number: int = 40) -> np.ndarray:
    """
    MFCCs (Mel-frequency cepstral coefficients) provide a small set of features
    which concisely describe the overall shape of a spectral envelope.
    In MIR, it is often used to describe timbre.
    """
    # print(f"Extracting MFCCs features from {filename}")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=number)
    return np.mean(mfccs.T, axis=0)


def extract_chroma(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Chroma features are an interesting and powerful representation for music audio
    in which the entire spectrum is projected onto 12 bins representing the 12
    distinct semitones (or chroma) of the musical octave.
    """
    # print(f"Extracting Chroma features from {filename}")
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    return np.mean(chroma.T, axis=0)


def extract_spec_contrast(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Spectral contrast is defined as the difference in amplitude between peaks and valleys in a sound spectrum.
    It provides a measure of spectral shape that has been shown to be important in the perception of timbre.
    """
    # print(f"Extracting Spectral Contrast features from {filename}")
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    return np.mean(spec_contrast.T, axis=0)


def extract_tonnetz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    The tonnetz is a representation of musical pitch classes that has been used to analyze harmony in Western tonal music.
    It can be useful for key detection and chord recognition.
    """
    # print(f"Extracting Tonnetz features from {filename}")
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    return np.mean(tonnetz.T, axis=0)


def extract_file_features_multi_threaded(file_path: str, cache_dir: str, threads=4, overwrite: bool = False) -> None:
    cache_file_name = os.path.basename(file_path)[:-4] + ".npy"

    if os.path.exists(os.path.join(cache_dir, cache_file_name)) and not overwrite:
        return

    else:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path)

        # Use a ThreadPoolExecutor to execute the functions in separate threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks to executor
            mfccs = executor.submit(extract_mfccs, audio, sample_rate, 40)
            chrom = executor.submit(extract_chroma, audio, sample_rate)
            spec_contrast = executor.submit(extract_spec_contrast, audio, sample_rate)
            tonnetz = executor.submit(extract_tonnetz, audio, sample_rate)

        # Concatenate all features into one array ORDER MATTERS
        features = np.concatenate([mfccs.result(), chrom.result(), spec_contrast.result(), tonnetz.result()])

        np.save(os.path.join(cache_dir, cache_file_name), features)


def load_features(audio_files_cache_dir: str) -> ([np.ndarray], [str]):
    data = []
    labels = []

    for speaker in os.listdir(audio_files_cache_dir):
        features = []
        for file in os.listdir(os.path.join(audio_files_cache_dir, speaker)):
            if file.endswith(".npy"):
                features.append(np.load(os.path.join(audio_files_cache_dir, speaker, file)))
        data.append(features)
    labels.append(speaker)

    return (data, labels)


def extract_features_multi_threaded(audio_files_dir: str, cache_dir: str, threads: int = 4) -> None:
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:

        for speaker in os.listdir(audio_files_dir):

            os.makedirs(os.path.join(cache_dir, speaker), exist_ok=True)

            speaker_files = os.listdir(os.path.join(audio_files_dir, speaker))
            for file in speaker_files:
                futures.append(
                    executor.submit(extract_file_features_multi_threaded, os.path.join(audio_files_dir, speaker, file), os.path.join(cache_dir, speaker), 4,
                                    False))

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Extracting features", dynamic_ncols=True,
                                colour="blue"):
            try:
                future.result()
            except Exception as e:
                print(e)


def TrainSupportVectorClassification(data, labels):
    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Encode labels to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    # Define the model
    model = SVC(probability=True, verbose=True)

    # Train the model and add a progress message
    print("Training model...")
    model.fit(X_train, y_train)

    # Test the model and add a progress message
    print("Testing model...")
    y_pred = model.predict(X_test)

    # Calculate the accuracy of our model and print it out with more information
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return (model, le)


def save_model(model, le):
    # Save the trained model and add a progress message
    print("Saving model...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save the label encoder and add a progress message
    print("Saving label encoder...")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)


def load_model(model_file, label_encoder_file):
    # Load the trained model and add a progress message
    print("Loading model...")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load the label encoder and add a progress message
    print("Loading label encoder...")
    with open(label_encoder_file, "rb") as f:
        le = pickle.load(f)

    # Now you can use `model.predict_proba()` to get prediction probabilities for new data
    return (model, le)


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
# def predict_speaker_with_probability(model, le, threads=4):
#     while True:
#         # Ask user for an audio file path and identify it.
#         audio_file_path = input("Please enter an audio file path or 'x' to close: ")
#         if audio_file_path.lower() == "x":
#             print("Closing the program.")
#             break
#         features = threads > 0 and extract_file_features_multi_threaded(audio_file_path, threads) or extract_file_features(
#             audio_file_path)
#         speaker_id = model.predict([features])
#         speaker_name = le.inverse_transform(speaker_id)
#
#         # Get the probability of the prediction
#         probability = model.predict_proba([features])
#         max_prob_index = np.argmax(probability)
#         max_prob = probability[0][max_prob_index]
#
#         print(f"The speaker is: {speaker_name[0]} with a probability of {max_prob * 100:.2f}%")
#
#         return speaker_name, probability


def save_data(data, labels):
    # Save the data and labels for future use
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)


def load_data():
    # Load the data and labels
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)


if __name__ == "__main__":
    import configuration
    import audio_tools as at

    conf = configuration.read_config()

    # at.convert_to_wav_multi_thread(use_conf=True, threads=15)
    # at.balance_audio_multi_thread(use_conf=True, threads=15)
    # at.normalize_audio_files_multi_thread(use_conf=True, threads=15, use_average_amplitude=True)
    # at.reduce_noise_multi_thread(threads=15, use_conf=True)

    # extract_features_multi_threaded(audio_files_dir=conf["Paths"]["training data"], cache_dir=conf["Paths"]["data cache"], threads=4)

    data, labels = load_features(conf["Paths"]["data cache"])

    # save_data(data, labels)

    # model, le = TrainSupportVectorClassification(data, labels)

    # save_model(model, le)

    # model , le = load_model("model_91.67.pkl" , "label_encoder_91.67.pkl")

    # predict_speaker_with_probability(model , le)
