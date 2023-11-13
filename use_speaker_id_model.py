import multi_thread_speaker_id as sid

model, le = sid.load_model("model.pkl", "label_encoder.pkl")

while True:
    sid.predict_speaker(model=model, le=le)
