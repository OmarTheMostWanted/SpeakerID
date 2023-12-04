import model_manager
import librosa.feature
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import concurrent.futures
import audio_feature_extraction as afe
import audio_normalizer as an
import audio_noise_reducer as anr


def TrainSupportVectorClassification(data, labels):
    # Encode labels to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=1
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


def save_model(model, le, accuracy: float, normv: float, speakers: [str] = None, model_dir: str = None, use_config: bool = True, balanced: bool = False,
               normalized: bool = False,
               denoised: bool = False,
               mfcc: bool = False, chroma: bool = False, spec_contrast: bool = False,
               tonnetz: bool = False, n_mfcc: int = None, ):
    # Save the trained model and add a progress message
    print("Saving model...")

    accuracy = np.round(accuracy, 2)

    if use_config:
        import configuration
        config = configuration.read_config()
        speakers = os.listdir(config["Paths"]["training data"])
        model_dir = config["Paths"]["models"]
        balanced = config.getboolean("Settings", "balance")
        normalized = config.getboolean("Settings", "normalize")
        denoised = config.getboolean("Settings", "reduce noise")
        mfcc = config.getboolean("Features", "mfcc")
        chroma = config.getboolean("Features", "chroma")
        spec_contrast = config.getboolean("Features", "spec contrast")
        tonnetz = config.getboolean("Features", "tonnetz")
        n_mfcc = config.getint("Settings", "n mfcc")

    model_obj = model_manager.ModelFile()
    model_obj.accuracy = accuracy
    model_obj.norm_val = normv
    model_obj.balanced = balanced
    model_obj.normalized = normalized
    model_obj.denoised = denoised
    model_obj.mfcc = mfcc
    model_obj.mfcc_val = n_mfcc
    model_obj.chroma = chroma
    model_obj.speccontrast = spec_contrast
    model_obj.tonnetz = tonnetz
    model_obj.speakers = speakers

    le_obj = model_manager.LabelEncoderFile()
    le_obj.accuracy = accuracy
    le_obj.norm_val = normv
    le_obj.balanced = balanced
    le_obj.normalized = normalized
    le_obj.denoised = denoised
    le_obj.mfcc = mfcc
    le_obj.mfcc_val = n_mfcc
    le_obj.chroma = chroma
    le_obj.speccontrast = spec_contrast
    le_obj.tonnetz = tonnetz
    le_obj.speakers = speakers

    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, model_obj.generate_model_name()), "wb") as f:
        pickle.dump(model, f)

    # Save the label encoder and add a progress message
    print("Saving label encoder...")
    with open(os.path.join(model_dir, le_obj.generate_le_name()), "wb") as f:
        pickle.dump(le, f)

    print(f"Model saved as {os.path.join(model_dir, model_obj.generate_model_name())}")
    print(f"Label Encoder saved as {os.path.join(model_dir, le_obj.generate_le_name())}")


def load_model(accuracy: float, normv: float, speakers: [str] = None, use_config: bool = True, model_dir: str = None, balanced: bool = False,
               normalized: bool = False,
               denoised: bool = False,
               mfcc: bool = False, chroma: bool = False, spec_contrast: bool = False,
               tonnetz: bool = False, n_mfcc: int = None, ):
    accuracy = np.round(accuracy, 2)

    if use_config:
        import configuration
        config = configuration.read_config()
        speakers = os.listdir(config["Paths"]["training data"])
        model_dir = config["Paths"]["models"]
        balanced = config.getboolean("Settings", "balance")
        normalized = config.getboolean("Settings", "normalize")
        denoised = config.getboolean("Settings", "reduce noise")
        mfcc = config.getboolean("Features", "mfcc")
        chroma = config.getboolean("Features", "chroma")
        spec_contrast = config.getboolean("Features", "spec contrast")
        tonnetz = config.getboolean("Features", "tonnetz")
        n_mfcc = config.getint("Settings", "n mfcc")

    model_obj = model_manager.ModelFile()
    model_obj.accuracy = accuracy
    model_obj.norm_val = normv
    model_obj.balanced = balanced
    model_obj.normalized = normalized
    model_obj.denoised = denoised
    model_obj.mfcc = mfcc
    model_obj.mfcc_val = n_mfcc
    model_obj.chroma = chroma
    model_obj.speccontrast = spec_contrast
    model_obj.tonnetz = tonnetz
    model_obj.speakers = speakers

    le_obj = model_manager.LabelEncoderFile()
    le_obj.accuracy = accuracy
    le_obj.norm_val = normv
    le_obj.balanced = balanced
    le_obj.normalized = normalized
    le_obj.denoised = denoised
    le_obj.mfcc = mfcc
    le_obj.mfcc_val = n_mfcc
    le_obj.chroma = chroma
    le_obj.speccontrast = spec_contrast
    le_obj.tonnetz = tonnetz
    le_obj.speakers = speakers

    print("Loading model...")
    with open(os.path.join(model_dir, model_obj.generate_model_name()), "rb") as f:
        model = pickle.load(f)

    # Load the label encoder and add a progress message
    print("Loading label encoder...")
    with open(os.path.join(model_dir, le_obj.generate_le_name()), "rb") as f:
        le = pickle.load(f)

    return model, le


def predict_speaker_with_probability(model, le, norv: float, threads=1):
    print(
        "Note that, we will assume that your model was created using the settings present in the config.ini file, if the model was trained used differnt"
        " settings expect bad accuracy or a crashes")

    import configuration
    config = configuration.read_config()

    while True:
        # Ask user for an audio file path and identify it.
        audio_file_path = input("Please enter an audio file path or 'x' to close: ")
        if audio_file_path.lower() == "x":
            print("Closing the program.")
            break

        if not audio_file_path.endswith(".wav"):
            print("only wav files are supported")
            continue

        file_name_t = audio_file_path

        if config.getboolean("Settings", "normalize"):
            print("normalizing file")
            file_name_t = "temp_normalized.wav"
            an.normalize_file(audio_file_path, "temp_normalized.wav", target_amplitude=norv)

        if config.getboolean("Settings", "reduce noise"):
            print("applying noise reduction")

            if config.getboolean("Settings", "normalize"):
                file_name_t = "temp_normalized_denoised.wav"
                anr.reduce_noise("temp_normalized.wav", output_path="temp_normalized_denoised.wav")
            else:
                file_name_t = "temp_denoised.wav"
                anr.reduce_noise(audio_file_path, output_path="temp_denoised.wav")

        audio, sample_rate = librosa.load(file_name_t)

        features = []

        extract_mfcc = config.getboolean("Features", "mfcc")
        nmfcc = config.getint("Settings", "N MFCC")
        extract_chroma = config.getboolean("Features", "chroma")
        extract_spec_contrast = config.getboolean("Features", "spec contrast")
        extract_tonnetz = config.getboolean("Features", "tonnetz")

        if extract_mfcc:
            features.append(afe.extract_mfccs_features(audio, sample_rate, nmfcc))
        if extract_chroma:
            features.append(afe.extract_chroma_features(audio, sample_rate))
        if extract_spec_contrast:
            features.append(afe.extract_spec_contrast_features(audio, sample_rate))
        if extract_tonnetz:
            features.append(afe.extract_tonnetz_features(audio, sample_rate))

        speaker_id = model.predict(np.concatenate(features))
        speaker_name = le.inverse_transform(speaker_id)

        # Get the probability of the prediction
        probability = model.predict_proba(np.concatenate(features))
        max_prob_index = np.argmax(probability)
        max_prob = probability[0][max_prob_index]

        if os.path.exists("temp_normalized.wav"):
            os.remove("temp_normalized.wav")
        if os.path.exists("temp_normalized_denoised.wav"):
            os.remove("temp_normalized_denoised.wav")
        if os.path.exists("temp_denoised.wav"):
            os.remove("temp_denoised.wav")

        print(f"The speaker is: {speaker_name[0]} with a probability of {max_prob * 100:.2f}%")


if __name__ == "__main__":
    # Example usage:
    # Creating an instance of AudioFeatureFile
    # audio_file_instance = AudioFeatureFile("John", "song", ["mfcc", "chroma"])
    #
    # # Generating a file name
    # file_name = audio_file_instance.get_file_name()
    # print(f"Generated file name: {file_name}")
    #
    # # Parsing a file name
    # parsed_audio_file = AudioFeatureFile("", "", [])
    # parsed_audio_file.parse_file_name(file_name)
    # print(f"Parsed speaker name: {parsed_audio_file.speaker_name}")
    # print(f"Parsed file name: {parsed_audio_file.file_name}")
    # print(f"Parsed features: {parsed_audio_file.features}")

    # Reading files for a given set of features and mfcc number
    speaker_directory = "/home/tmw/Digivox/data_cache/20319"
    features_to_find = ["speccontrast"]

    matching_files = AudioFeatureFile.read_files_for_features(speaker_directory, features_to_find, None)
    print(f"Matching files: {matching_files}")

    model, le = load_model(
        "/home/tmw/Code/SpeakerID", "model_100.0_.pkl", "label_encoder_100.0_.pkl"
    )

    predict_speaker_with_probability(model, le)
