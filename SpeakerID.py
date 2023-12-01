import os

import AudioFile
import audio_wav_converter as aw
import audio_balancer as ab
import audio_normalizer as an
import audio_noise_reducer as anr
import audio_feature_extraction as afe
import multi_thread_speaker_id as sid
import configuration

amplitude = -27.35

config = configuration.read_config()
input_dir = config["Paths"]["training data"]
data_dir = config["Paths"]["feature data"]
over_write = config.getboolean("Settings", "overwrite data")

extract_mfcc = config.getboolean("Features", "mfcc")
nmfcc = config.getint("Settings", "N MFCC")
extract_chroma = config.getboolean("Features", "chroma")
extract_spec_contrast = config.getboolean("Features", "spec contrast")
extract_tonnetz = config.getboolean("Features", "tonnetz")

file = "/home/tmw/Digivox/data_denoised/20464/00185881_000_balanced_normalized(-26.89)_denoised.wav"

afe.extract_file_features_multi_threaded(file, data_dir, 4, False, extract_mfcc, extract_chroma, extract_spec_contrast, extract_tonnetz, nmfcc)

afe.extract_file_features_multi_threaded(file, data_dir, 4, False, extract_mfcc, extract_chroma, extract_spec_contrast, extract_tonnetz, nmfcc)


# aw.convert_to_wav_multi_thread(threads=8)
# ab.balance_audio_multi_thread(threads=8)
# amplitude = an.normalize_audio_files_multi_thread(threads=8)
# anr.reduce_noise_multi_thread(threads=6)
# afe.extract_features_multi_threaded(threads=2)
# data, labels = afe.load_features(amplitude)
#
# model, le, accuracy = sid.TrainSupportVectorClassification(data, labels)
#
# sid.save_model(model, le, accuracy, amplitude)
#
# model, le = sid.load_model(accuracy, amplitude)
#
# sid.predict_speaker_with_probability(model, le, amplitude, 4)
