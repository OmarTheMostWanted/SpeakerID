import audio_wav_converter as aw
import audio_balancer as ab
import audio_normalizer as an
import audio_noise_reducer as anr
import audio_feature_extraction as afe
import multi_thread_speaker_id as sid
import audio_remove_silence as ars
import configuration

import pandas as pd
import numpy as np
import librosa

# config = configuration.read_config()
# input_dir = config["Paths"]["training data"]
# data_dir = config["Paths"]["feature data"]
# over_write = config.getboolean("Settings", "overwrite data")
#
# extract_mfcc = config.getboolean("Features", "mfcc")
# nmfcc = config.getint("Settings", "N MFCC")
# extract_chroma = config.getboolean("Features", "chroma")
# extract_spec_contrast = config.getboolean("Features", "spec contrast")
# extract_tonnetz = config.getboolean("Features", "tonnetz")


#  ues pandas, for plotting and visualizing
# get more data
# seperate the features
# look at the phonexia documentation

# lime and sub: lime: take a data point, change it slightly, and see how it effects the model

# CNNs as a next step.


import librosa

import configuration

config = configuration.read_config()

audio_file_path = "/home/tmw/Digivox/audio_data/data/20319/00210632_000.WAV"
file_name_t = audio_file_path

if os.path.exists("temp_normalized.wav"):
    os.remove("temp_normalized.wav")
if os.path.exists("temp_denoised.wav"):
    os.remove("temp_denoised.wav")
if os.path.exists("temp_removed_silence.wav"):
    os.remove("temp_removed_silence.wav")

anr.reduce_noise(file_name_t, output_path="temp_denoised.wav")
file_name_t = "temp_denoised.wav"

print("Removing silence")
ars.remove_silence_from_audio_librosa(file_name_t, "temp_removed_silence.wav")
file_name_t = "temp_removed_silence.wav"

print("normalizing file")
an.normalize_file(audio_file_path, "temp_normalized.wav", target_amplitude=-20)
file_name_t = "temp_normalized.wav"

audio, sample_rate = librosa.load(file_name_t)

features = []

extract_mfcc = config.getboolean("Features", "mfcc")
nmfcc = config.getint("Settings", "N MFCC")
extract_chroma = config.getboolean("Features", "chroma")
extract_spec_contrast = config.getboolean("Features", "spec contrast")
extract_tonnetz = config.getboolean("Features", "tonnetz")


# mfcc, chroma, spec_constrast, tonnetz


def extract_features(file_name, label):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = afe.extract_mfccs_features(audio, sample_rate, nmfcc)
        chroma = afe.extract_chroma_features(audio, sample_rate)
        spec_contrast = afe.extract_spec_contrast_features(audio, sample_rate)
        tonnetz = afe.extract_tonnetz_features(audio, sample_rate)

        # Create a pandas Series for this file
        features = pd.DataFrame({
            'mfccs': mfccs,
            'chroma': chroma,
            'spec_contrast': spec_contrast,
            'tonnetz': tonnetz,
            'label': label
        })

        return features

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None


features = extract_features('audio.wav', 'label')

from sklearn import svm
from sklearn.model_selection import train_test_split

# Assuming 'data' is a DataFrame containing your extracted features and labels
X = features.drop('label', axis=1)  # Features
y = features['label']  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a SVC classifier
clf = svm.SVC()

# Train the model
clf.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred = clf.predict(X_test)

print("end")

# aw.convert_to_wav_multi_thread(threads=8)
# anr.reduce_noise_multi_thread(threads=6)
# ars.remove_silence_multi_thread(threads=8)
# selected = ab.balance_audio_multi_thread(threads=8)
# amplitude = an.normalize_audio_files_multi_thread(threads=8, selected=selected)
# afe.extract_features_multi_threaded(threads=2, selected=selected)
# data, labels = afe.load_features(-26, selected=selected)
#
# model, le, accuracy = sid.TrainSupportVectorClassification(data, labels)
#
# sid.save_model(model, le, accuracy, amplitude)
#
# model, le = sid.load_model(accuracy, amplitude)
#
# sid.predict_speaker_with_probability(model, le, amplitude, 4)
