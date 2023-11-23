import audio_tools as at
import multi_thread_speaker_id as sid

model, le = sid.load_model("/home/tmw/Code/SpeakerID", "/home/tmw/Code/SpeakerID/model_100.0_.pkl", "label_encoder_100.0_.pkl")

sid.predict_speaker_with_probability(model, le)