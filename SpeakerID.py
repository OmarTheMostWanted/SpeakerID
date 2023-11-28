import os

import AudioFile
import audio_wav_converter as aw
import audio_balancer as ab
import audio_normalizer as an
import audio_noise_reducer as anr
import audio_feature_extraction as afe
import multi_thread_speaker_id as sid

amplitude = -27.35

aw.convert_to_wav_multi_thread(threads=8)
ab.balance_audio_multi_thread(threads=8)
amplitude = an.normalize_audio_files_multi_thread(threads=8)
anr.reduce_noise_multi_thread(threads=6)
afe.extract_features_multi_threaded(threads=4)
data, labels = afe.load_features(amplitude)

model, le, accuracy = sid.TrainSupportVectorClassification(data, labels)

sid.save_model(model, le, accuracy, amplitude)

model, le = sid.load_model(accuracy, amplitude)

sid.predict_speaker_with_probability(model, le, amplitude, 4)
