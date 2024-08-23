import model_manager
import librosa.feature
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import audio_feature_extraction as afe
import audio_normalizer as an
import audio_noise_reducer as anr
import audio_remove_silence as asr
import audio_split


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


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculate the accuracy of our model and print it out with more information
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return (model, le, accuracy)


def save_model(model, le, accuracy: float, normv: float, speakers: [str] = None, model_dir: str = None, use_config: bool = True, denoised: bool = False,
               desilence: bool = False, balanced: bool = False, normalized: bool = False, mfcc: bool = False, chroma: bool = False, spec_contrast: bool = False,
               tonnetz: bool = False, n_mfcc: int = None):

    accuracy = np.round(accuracy, 2)

    if use_config:
        import configuration
        config = configuration.read_config()
        speakers = os.listdir(config["Paths"]["training data"])
        model_dir = config["Paths"]["models"]
        balanced = config.getboolean("Settings", "balance")
        normalized = config.getboolean("Settings", "normalize")
        denoised = config.getboolean("Settings", "reduce noise")
        desilence = config.getboolean("Settings", "remove silence")
        mfcc = config.getboolean("Features", "mfcc")
        chroma = config.getboolean("Features", "chroma")
        spec_contrast = config.getboolean("Features", "spec contrast")
        tonnetz = config.getboolean("Features", "tonnetz")
        n_mfcc = config.getint("Settings", "n mfcc")

    model_obj = model_manager.ModelFile()
    model_obj.accuracy = accuracy
    model_obj.norm_val = normv
    model_obj.desilenced = desilence
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
    le_obj.desilenced = desilence
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

    with open(os.path.join(model_dir, le_obj.generate_le_name()), "wb") as f:
        pickle.dump(le, f)

    print(f"Model saved as {os.path.join(model_dir, model_obj.generate_model_name())}")
    print(f"Label Encoder saved as {os.path.join(model_dir, le_obj.generate_le_name())}")


def load_model(accuracy: float, normv: float, speakers: [str] = None, use_config: bool = True, model_dir: str = None, denoised: bool = False,
               desilence: bool = False, balanced: bool = False, normalized: bool = False, mfcc: bool = False, chroma: bool = False, spec_contrast: bool = False,
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
        desilence = config.getboolean("Settings", "remove silence")

    model_obj = model_manager.ModelFile()
    model_obj.accuracy = accuracy
    model_obj.norm_val = normv
    model_obj.desilenced = desilence
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
    le_obj.desilenced = desilence
    le_obj.balanced = balanced
    le_obj.normalized = normalized
    le_obj.denoised = denoised
    le_obj.mfcc = mfcc
    le_obj.mfcc_val = n_mfcc
    le_obj.chroma = chroma
    le_obj.speccontrast = spec_contrast
    le_obj.tonnetz = tonnetz
    le_obj.speakers = speakers

    with open(os.path.join(model_dir, model_obj.generate_model_name()), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(model_dir, le_obj.generate_le_name()), "rb") as f:
        le = pickle.load(f)

    return model, le


def predict_speaker_with_probability(model, le, norv: float, threads=1):
    print(
        "Note that, we will assume that your model was created using the settings present in the config.ini file, if the model was trained using different"
        " settings expect bad accuracy or a crashes")

    import configuration
    config = configuration.read_config()

    while True:
        # Ask user for an audio file path and identify it.
        audio_file_path = input("Please enter an audio file path or 'x' to close: ")
        if audio_file_path.lower() == "x":
            print("Closing the program.")
            break

        if not os.path.isfile(audio_file_path) and not audio_file_path[:-4].lower() == ".wav":
            print("only wav files are supported")
            continue

        file_name_t = audio_file_path

        if os.path.exists("temp_normalized.wav"):
            os.remove("temp_normalized.wav")
        if os.path.exists("temp_denoised.wav"):
            os.remove("temp_denoised.wav")
        if os.path.exists("temp_removed_silence.wav"):
            os.remove("temp_removed_silence.wav")

        if config.getboolean("Settings", "reduce noise"):
            print("applying noise reduction")
            anr.reduce_noise(file_name_t, output_path="temp_denoised.wav")
            file_name_t = "temp_denoised.wav"

        if config.getboolean("Settings", "remove silence"):
            print("Removing silence")
            asr.remove_silence_from_audio_librosa(file_name_t, "temp_removed_silence.wav")
            file_name_t = "temp_removed_silence.wav"

        if config.getboolean("Settings", "normalize"):
            print("normalizing file")
            an.normalize_file(audio_file_path, "temp_normalized.wav", target_amplitude=norv)
            file_name_t = "temp_normalized.wav"

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

        concatenated = [np.concatenate(features)]

        speaker_id = model.predict(concatenated)
        speaker_name = le.inverse_transform(speaker_id)

        # Get the probability of the prediction
        probability = model.predict_proba(concatenated)
        max_prob_index = np.argmax(probability)
        max_prob = probability[0][max_prob_index]

        if os.path.exists("temp_normalized.wav"):
            os.remove("temp_normalized.wav")
        if os.path.exists("temp_denoised.wav"):
            os.remove("temp_denoised.wav")
        if os.path.exists("temp_removed_silence.wav"):
            os.remove("temp_removed_silence.wav")

        print(f"The speaker is: {speaker_name[0]} with a probability of {max_prob * 100:.2f}%")


def predict_speaker_with_combined_probability(model_mfcc, model_chroma, model_spec_contrast, model_tonnetz, le, norv: float, threads=4):
    print(
        "Note that, we will assume that your model was created using the settings present in the config.ini file, if the model was trained using different"
        " settings expect bad accuracy or a crashes")

    import configuration
    config = configuration.read_config()

    while True:
        # Ask user for an audio file path and identify it.
        audio_file_path = input("Please enter an audio file path or 'x' to close: ")
        if audio_file_path.lower() == "x":
            print("Closing the program.")
            break

        if not os.path.isfile(audio_file_path) and not audio_file_path[:-4].lower() == ".wav":
            print("only wav files are supported")
            continue

        file_name_t = audio_file_path

        if os.path.exists("temp_normalized.wav"):
            os.remove("temp_normalized.wav")
        if os.path.exists("temp_denoised.wav"):
            os.remove("temp_denoised.wav")
        if os.path.exists("temp_removed_silence.wav"):
            os.remove("temp_removed_silence.wav")

        if config.getboolean("Settings", "reduce noise"):
            print("applying noise reduction")
            anr.reduce_noise(file_name_t, output_path="temp_denoised.wav")
            file_name_t = "temp_denoised.wav"

        if config.getboolean("Settings", "remove silence"):
            print("Removing silence")
            asr.remove_silence_from_audio_librosa(file_name_t, "temp_removed_silence.wav")
            file_name_t = "temp_removed_silence.wav"

        if config.getboolean("Settings", "normalize"):
            print("normalizing file")
            an.normalize_file(audio_file_path, "temp_normalized.wav", target_amplitude=norv)
            file_name_t = "temp_normalized.wav"

        audio, sample_rate = librosa.load(file_name_t)

        nmfcc = config.getint("Settings", "N MFCC")

        mfcc_features = afe.extract_mfccs_features(audio, sample_rate, nmfcc)
        chroma_features = afe.extract_chroma_features(audio, sample_rate)
        spec_contrast_features = afe.extract_spec_contrast_features(audio, sample_rate)
        tonnetz_features = afe.extract_tonnetz_features(audio, sample_rate)

        speaker_id_mfcc = model_mfcc.predict([mfcc_features])
        speaker_name_mfcc = le.inverse_transform(speaker_id_mfcc)
        probability_mfcc = model_mfcc.predict_proba([mfcc_features])
        max_prob_index_mfcc = np.argmax(probability_mfcc)
        max_prob_mfcc = probability_mfcc[0][max_prob_index_mfcc]
        print(f"Model MFCC thinks the speaker is: {speaker_name_mfcc[0]} with a probability of {max_prob_mfcc * 100:.2f}%")

        speaker_id_chroma = model_chroma.predict([chroma_features])
        speaker_name_chroma = le.inverse_transform(speaker_id_chroma)
        probability_chroma = model_chroma.predict_proba([chroma_features])
        max_prob_index_chroma = np.argmax(probability_chroma)
        max_prob_chroma = probability_chroma[0][max_prob_index_chroma]
        print(f"Model Chroma thinks the speaker is: {speaker_name_chroma[0]} with a probability of {max_prob_chroma * 100:.2f}%")

        speaker_id_spec_contrast = model_spec_contrast.predict([spec_contrast_features])
        speaker_name_spec_contrast = le.inverse_transform(speaker_id_spec_contrast)
        probability_spec_contrast = model_spec_contrast.predict_proba([spec_contrast_features])
        max_prob_index_spec_contrast = np.argmax(probability_spec_contrast)
        max_prob_spec_contrast = probability_spec_contrast[0][max_prob_index_spec_contrast]
        print(f"Model Spec Contrast thinks the speaker is: {speaker_name_spec_contrast[0]} with a probability of {max_prob_spec_contrast * 100:.2f}%")

        speaker_id_tonnetz = model_tonnetz.predict([tonnetz_features])
        speaker_name_tonnetz = le.inverse_transform(speaker_id_tonnetz)
        probability_tonnetz = model_tonnetz.predict_proba([tonnetz_features])
        max_prob_index_tonnetz = np.argmax(probability_tonnetz)
        max_prob_tonnetz = probability_tonnetz[0][max_prob_index_tonnetz]
        print(f"Model Tonnetz thinks the speaker is: {speaker_name_tonnetz[0]} with a probability of {max_prob_tonnetz * 100:.2f}%")

        speaker_ids = [speaker_id_mfcc[0], speaker_id_chroma[0], speaker_id_spec_contrast[0], speaker_id_tonnetz[0]]
        probabilities = [max_prob_mfcc, max_prob_chroma, max_prob_spec_contrast, max_prob_tonnetz]

        combined_speaker_id = max(set(speaker_ids), key=speaker_ids.count)
        combined_speaker_name = le.inverse_transform([combined_speaker_id])
        combined_probability = sum(probabilities) / len(probabilities)

        if os.path.exists("temp_normalized.wav"):
            os.remove("temp_normalized.wav")
        if os.path.exists("temp_denoised.wav"):
            os.remove("temp_denoised.wav")
        if os.path.exists("temp_removed_silence.wav"):
            os.remove("temp_removed_silence.wav")



        print(f"The combined result is: \033[1;32;40m {combined_speaker_name[0]} \033[m  with a probability of {combined_probability * 100:.2f}%")


def predict_speaker_with_combined_split_probability(model_mfcc, model_chroma, model_spec_contrast, model_tonnetz, le, norv: float, threads=1):
    print(
        "Note that, we will assume that your model was created using the settings present in the config.ini file, if the model was trained using different"
        " settings expect bad accuracy or a crashes")

    import configuration
    config = configuration.read_config()

    while True:
        # Ask user for an audio file path and identify it.
        audio_file_path = input("Please enter an audio file path or 'x' to close: ")
        if audio_file_path.lower() == "x":
            print("Closing the program.")
            break

        if not os.path.isfile(audio_file_path) and not audio_file_path[:-4].lower() == ".wav":
            print("only wav files are supported")
            continue

        file_name_t = audio_file_path

        if os.path.exists("temp_normalized.wav"):
            os.remove("temp_normalized.wav")
        if os.path.exists("temp_denoised.wav"):
            os.remove("temp_denoised.wav")
        if os.path.exists("temp_removed_silence.wav"):
            os.remove("temp_removed_silence.wav")

        if config.getboolean("Settings", "reduce noise"):
            print("applying noise reduction")
            anr.reduce_noise(file_name_t, output_path="temp_denoised.wav")
            file_name_t = "temp_denoised.wav"

        if config.getboolean("Settings", "remove silence"):
            print("Removing silence")
            asr.remove_silence_from_audio_librosa(file_name_t, "temp_removed_silence.wav")
            file_name_t = "temp_removed_silence.wav"

        if config.getboolean("Settings", "normalize"):
            print("normalizing file")
            an.normalize_file(audio_file_path, "temp_normalized.wav", target_amplitude=norv)
            file_name_t = "temp_normalized.wav"

        _, audio_files = audio_split.split_audio_librosa(file_name_t, "temp_split_", split_minutes=5)

        nmfcc = config.getint("Settings", "N MFCC")

        speaker_ids = []
        probabilities = []

        for audio_file in audio_files:
            audio, sample_rate = librosa.load(audio_file)

            mfcc_features = afe.extract_mfccs_features(audio, sample_rate, nmfcc)
            chroma_features = afe.extract_chroma_features(audio, sample_rate)
            spec_contrast_features = afe.extract_spec_contrast_features(audio, sample_rate)
            tonnetz_features = afe.extract_tonnetz_features(audio, sample_rate)

            speaker_id_mfcc = model_mfcc.predict([mfcc_features])
            speaker_name_mfcc = le.inverse_transform(speaker_id_mfcc)
            probability_mfcc = model_mfcc.predict_proba([mfcc_features])
            max_prob_index_mfcc = np.argmax(probability_mfcc)
            max_prob_mfcc = probability_mfcc[0][max_prob_index_mfcc]
            print(f"Model MFCC thinks the speaker is: {speaker_name_mfcc[0]} with a probability of {max_prob_mfcc * 100:.2f}%")

            speaker_id_chroma = model_chroma.predict([chroma_features])
            speaker_name_chroma = le.inverse_transform(speaker_id_chroma)
            probability_chroma = model_chroma.predict_proba([chroma_features])
            max_prob_index_chroma = np.argmax(probability_chroma)
            max_prob_chroma = probability_chroma[0][max_prob_index_chroma]
            print(f"Model Chroma thinks the speaker is: {speaker_name_chroma[0]} with a probability of {max_prob_chroma * 100:.2f}%")

            speaker_id_spec_contrast = model_spec_contrast.predict([spec_contrast_features])
            speaker_name_spec_contrast = le.inverse_transform(speaker_id_spec_contrast)
            probability_spec_contrast = model_spec_contrast.predict_proba([spec_contrast_features])
            max_prob_index_spec_contrast = np.argmax(probability_spec_contrast)
            max_prob_spec_contrast = probability_spec_contrast[0][max_prob_index_spec_contrast]
            print(f"Model Spec Contrast thinks the speaker is: {speaker_name_spec_contrast[0]} with a probability of {max_prob_spec_contrast * 100:.2f}%")

            speaker_id_tonnetz = model_tonnetz.predict([tonnetz_features])
            speaker_name_tonnetz = le.inverse_transform(speaker_id_tonnetz)
            probability_tonnetz = model_tonnetz.predict_proba([tonnetz_features])
            max_prob_index_tonnetz = np.argmax(probability_tonnetz)
            max_prob_tonnetz = probability_tonnetz[0][max_prob_index_tonnetz]
            print(f"Model Tonnetz thinks the speaker is: {speaker_name_tonnetz[0]} with a probability of {max_prob_tonnetz * 100:.2f}%")

            speaker_ids.extend([speaker_id_mfcc[0], speaker_id_chroma[0], speaker_id_spec_contrast[0], speaker_id_tonnetz[0]])
            probabilities.extend([max_prob_mfcc, max_prob_chroma, max_prob_spec_contrast, max_prob_tonnetz])

            os.remove(audio_file)

        combined_speaker_id = max(set(speaker_ids), key=speaker_ids.count)
        combined_speaker_name = le.inverse_transform([combined_speaker_id])
        combined_probability = sum(probabilities) / len(probabilities)

        if os.path.exists("temp_normalized.wav"):
            os.remove("temp_normalized.wav")
        if os.path.exists("temp_denoised.wav"):
            os.remove("temp_denoised.wav")
        if os.path.exists("temp_removed_silence.wav"):
            os.remove("temp_removed_silence.wav")

        print(f"The combined result is: {combined_speaker_name[0]} with a probability of {combined_probability * 100:.2f}%")


if __name__ == "__main__":
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
    desilence = config.getboolean("Settings", "remove silence")

    model_obj = model_manager.ModelFile()
    model_obj.accuracy = 69.0
    model_obj.norm_val = 20.2
    model_obj.desilenced = desilence
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
    le_obj.accuracy = 69.0
    le_obj.norm_val = 20.2
    le_obj.desilenced = desilence
    le_obj.balanced = balanced
    le_obj.normalized = normalized
    le_obj.denoised = denoised
    le_obj.mfcc = mfcc
    le_obj.mfcc_val = n_mfcc
    le_obj.chroma = chroma
    le_obj.speccontrast = spec_contrast
    le_obj.tonnetz = tonnetz
    le_obj.speakers = speakers

    model = model_obj.generate_model_name()
    le = le_obj.generate_le_name()

    print(model)
    print(le)

    model2 = model_manager.ModelFile.from_name(model)
    le2 = model_manager.LabelEncoderFile.from_name(le)

    print(model2.generate_model_name())
    print(le2.generate_le_name())
