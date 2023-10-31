import sys
import json
import difflib
from vosk import Model, KaldiRecognizer
import os

def transcribe_audio(audio_path, language_model_path):
    model = Model(model_path=language_model_path)
    rec = KaldiRecognizer(model, 16000)

    with open(audio_path, 'rb') as f:
        while True:
            data = f.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                pass

    result = json.loads(rec.FinalResult())
    return result['text']

def compare_transcriptions(generated_transcription, transcription_file_path):
    with open(transcription_file_path, 'r') as f:
        original_transcription = f.read()

    s = difflib.SequenceMatcher(None, generated_transcription, original_transcription)
    return s.ratio() * 100

if __name__ == "__main__":
    if len(sys.argv) == 3:
        audio_file_path = sys.argv[1]
        language_model_path = sys.argv[2]
        generated_transcription = transcribe_audio(audio_file_path, language_model_path)
        #write to file
        with open(f"transcription of {os.path.basename(audio_file_path)}.txt", "w") as f:
            f.write(generated_transcription)
            print(f"Generated Transcription: {generated_transcription}")
            sys.exit(0)

    elif len(sys.argv) == 4:
        audio_file_path = sys.argv[1]
        language_model_path = sys.argv[2]
        transcription_file_path = sys.argv[3]
        generated_transcription = transcribe_audio(audio_file_path, language_model_path)
        score = compare_transcriptions(generated_transcription , transcription_file_path)
        print(f"Transcritopn acuracy {score}")

    else:
        print("Please provide audio file path, language model path, and optionally transcription file path")
        sys.exit(1)

    