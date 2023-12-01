import os

import AudioFile
import audio_wav_converter as aw
import audio_balancer as ab
import audio_normalizer as an
import audio_noise_reducer as anr
import audio_feature_extraction as afe
import multi_thread_speaker_id as sid
import configuration



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

aw.convert_to_wav_multi_thread(threads=8)
selected = ab.balance_audio_multi_thread(threads=8)
anr.reduce_noise_multi_thread(threads=6, selected=selected)
amplitude = an.normalize_audio_files_multi_thread(threads=8, selected=selected)
afe.extract_features_multi_threaded(threads=2 , selected=selected)
data, labels = afe.load_features(-26 , selected=selected)
#
model, le, accuracy = sid.TrainSupportVectorClassification(data, labels)

sid.save_model(model, le, accuracy, amplitude)

model, le = sid.load_model(accuracy, amplitude)

sid.predict_speaker_with_probability(model, le, amplitude, 4)
