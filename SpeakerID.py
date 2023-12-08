import audio_wav_converter as aw
import audio_balancer as ab
import audio_normalizer as an
import audio_noise_reducer as anr
import audio_feature_extraction as afe
import multi_thread_speaker_id as sid
import audio_remove_silence as ars
import audio_split
import configuration
import os

config = configuration.read_config()

aw.convert_to_wav_multi_thread(threads=8)
anr.reduce_noise_multi_thread(threads=6)
ars.remove_silence_multi_thread(threads=8)
audio_split.split_audio_multi_thread(threads=8)
selected = ab.balance_audio_multi_thread(threads=8)
amplitude = an.normalize_audio_files_multi_thread(threads=8, selected=selected)

afe.extract_all_features_multi_threaded(use_config=True, threads=6, over_write=False, selected=selected)

data_mfcc, labels_mfcc = afe.load_features(normv=amplitude, use_config=False, audio_data_dir=config["Paths"]["feature data"], balanced=True, denoised=True,
                                           n_mfcc=config.getint("Settings", "n mfcc"),
                                           normalized=True, split=True, mfcc=True , selected=selected)
data_chroma, labels_chroma = afe.load_features(normv=amplitude, use_config=False, audio_data_dir=config["Paths"]["feature data"], balanced=True, denoised=True,
                                               n_mfcc=config.getint("Settings", "n mfcc"),
                                               normalized=True, split=True, chroma=True, selected=selected)
data_spec_contrast, labels_spec_contrast = afe.load_features(normv=amplitude, use_config=False, audio_data_dir=config["Paths"]["feature data"], balanced=True,
                                                             denoised=True, n_mfcc=config.getint("Settings", "n mfcc"),
                                                             normalized=True, split=True, spec_contrast=True, selected=selected)
data_tonnetz, labels_tonnetz = afe.load_features(normv=amplitude, use_config=False, audio_data_dir=config["Paths"]["feature data"], balanced=True,
                                                 denoised=True, n_mfcc=config.getint("Settings", "n mfcc"),
                                                 normalized=True, split=True, tonnetz=True, selected=selected)

model_mfcc, le_mfcc, accuracy_mfcc = sid.TrainSupportVectorClassification(data_mfcc, labels_mfcc)
model_chroma, le_chroma, accuracy_chroma = sid.TrainSupportVectorClassification(data_chroma, labels_chroma)
model_spec_contrast, le_spec_contrast, accuracy_spec_contrast = sid.TrainSupportVectorClassification(data_spec_contrast, labels_spec_contrast)
model_tonnetz, le_tonnetz, accuracy_tonnetz = sid.TrainSupportVectorClassification(data_tonnetz, labels_tonnetz)

sid.save_model(model_mfcc, le_mfcc, accuracy_mfcc, amplitude, use_config=False, mfcc=True, speakers=os.listdir(config["Paths"]["training data"]), denoised=True,
               normalized=True, balanced=True, model_dir=config["Paths"]["models"], n_mfcc=config.getint("Settings", "n mfcc"))
sid.save_model(model_chroma, le_chroma, accuracy_chroma, amplitude, use_config=False, chroma=True, speakers=os.listdir(config["Paths"]["training data"]),
               denoised=True, normalized=True, balanced=True, model_dir=config["Paths"]["models"])
sid.save_model(model_spec_contrast, le_spec_contrast, accuracy_spec_contrast, amplitude, use_config=False, spec_contrast=True,
               speakers=os.listdir(config["Paths"]["training data"]), denoised=True, normalized=True, balanced=True, model_dir=config["Paths"]["models"])
sid.save_model(model_tonnetz, le_tonnetz, accuracy_tonnetz, amplitude, use_config=False, tonnetz=True, speakers=os.listdir(config["Paths"]["training data"]),
               denoised=True, normalized=True, balanced=True, model_dir=config["Paths"]["models"])

model_mfcc_l, le_mfcc_l = sid.load_model(accuracy=accuracy_mfcc, normv=amplitude, use_config=False, mfcc=True,
                                         speakers=os.listdir(config["Paths"]["training data"]), denoised=True, normalized=True, balanced=True,
                                         model_dir=config["Paths"]["models"], n_mfcc=config.getint("Settings", "n mfcc"))
model_chroma_l, le_chroma_l = sid.load_model(accuracy=accuracy_chroma, normv=amplitude, use_config=False, chroma=True,
                                             speakers=os.listdir(config["Paths"]["training data"]),
                                             denoised=True, normalized=True, balanced=True, model_dir=config["Paths"]["models"], )
model_spec_contrast_l, le_spec_contrast_l = sid.load_model(accuracy=accuracy_spec_contrast, normv=amplitude, use_config=False, spec_contrast=True,
                                                           speakers=os.listdir(config["Paths"]["training data"]), denoised=True,
                                                           normalized=True, balanced=True, model_dir=config["Paths"]["models"])
model_tonnetz_l, le_tonnetz_l = sid.load_model(accuracy=accuracy_tonnetz, normv=amplitude, use_config=False, tonnetz=True,
                                               speakers=os.listdir(config["Paths"]["training data"]), denoised=True, normalized=True,
                                               balanced=True, model_dir=config["Paths"]["models"])

sid.predict_speaker_with_combined_probability(model_mfcc_l, model_chroma_l, model_spec_contrast_l, model_tonnetz_l, le_mfcc_l, amplitude)

# amplitude = -30.0
#
#
# model_mfcc_l, le_mfcc_l = sid.load_model(accuracy=0.21, normv=amplitude, use_config=False, mfcc=True,
#                                          speakers=os.listdir(config["Paths"]["training data"]), denoised=True, normalized=True, balanced=True,
#                                          model_dir=config["Paths"]["models"], n_mfcc=config.getint("Settings", "n mfcc"))
# model_chroma_l, le_chroma_l = sid.load_model(accuracy=1.0, normv=amplitude, use_config=False, chroma=True,
#                                              speakers=os.listdir(config["Paths"]["training data"]),
#                                              denoised=True, normalized=True, balanced=True, model_dir=config["Paths"]["models"], )
# model_spec_contrast_l, le_spec_contrast_l = sid.load_model(accuracy=1.0, normv=amplitude, use_config=False, spec_contrast=True,
#                                                            speakers=os.listdir(config["Paths"]["training data"]), denoised=True,
#                                                            normalized=True, balanced=True, model_dir=config["Paths"]["models"])
# model_tonnetz_l, le_tonnetz_l = sid.load_model(accuracy=0.97, normv=amplitude, use_config=False, tonnetz=True,
#                                                speakers=os.listdir(config["Paths"]["training data"]), denoised=True, normalized=True,
#                                                balanced=True, model_dir=config["Paths"]["models"])
#
# sid.predict_speaker_with_combined_probability(model_mfcc_l, model_chroma_l, model_spec_contrast_l, model_tonnetz_l, le_mfcc_l, amplitude)