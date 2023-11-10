import difflib
import os
import sys
import json
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer


def make_chunks(audio, chunk_length_ms):
    """
    Breaks up the audio file into chunks each of length chunk_length_ms
    (in milliseconds)
    """
    chunk_length_ms = chunk_length_ms  # pydub calculates in millisec
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):  # Make chunks of chunk_length_ms sec
        chunks.append(audio[i:i + chunk_length_ms])
    return chunks


def transcribe_audio(audio_path, model_path):
    model = Model(model_path)
    rec = KaldiRecognizer(model, 8000)

    with open(audio_path, 'rb') as f:
        while True:
            data = f.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                pass

    result = json.loads(rec.FinalResult())
    return result['text']


def split_audio(audio_path, chunk_length_ms=60000):
    audio = AudioSegment.from_file(audio_path)

    chunks = make_chunks(audio, chunk_length_ms)  # Make chunks of one sec

    # Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")

    return [chunk_name for chunk_name in os.listdir() if chunk_name.startswith('chunk')]


def main():
    audio_path = sys.argv[1]
    model_path = sys.argv[2]

    chunk_paths = split_audio(audio_path)

    with open('transcriptions.txt', 'w') as f:
        for chunk_path in chunk_paths:
            transcription = transcribe_audio(chunk_path, model_path)
            f.write(transcription + '\n')


if __name__ == '__main__':
    main()


def transcribe_audio(audio_path):
    model = Model(lang="es")
    rec = KaldiRecognizer(model, 8000)

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

    main()
    quit()


    if len(sys.argv) == 3:
        audio_file_path = sys.argv[1]
        language_model_path = sys.argv[2]
        generated_transcription = transcribe_audio(audio_file_path, language_model_path)
        # write to file
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
        print(f"Transcription acuracy {score}")

    else:
        print("Please provide audio file path, language model path, and optionally transcription file path")
        sys.exit(1)

    from kaldi.asr import NnetLatticeFasterRecognizer
    from kaldi.decoder import LatticeFasterDecoder
    from kaldi.nnet3 import NnetSimpleComputationOptions
    from kaldi.matrix import Matrix
    from kaldi.util.table import SequentialWaveReader
    from kaldi.feat.mfcc import Mfcc, MfccOptions
    from kaldi.feat.functions import compute_deltas, DeltaFeaturesOptions
    from kaldi.transform.cmvn import Cmvn
    from kaldi.matrix import SubVector
    from kaldi.util.table import SequentialBaseFloatMatrixReader

    # Paths to the model files
    model_path = "/path/to/model"
    graph_path = "/path/to/graph"
    symbols_path = "/path/to/symbols"
    features_rspecifier = "/path/to/features"

    # Initialize the recognizer
    decoder_opts = LatticeFasterDecoder.Options()
    decodable_opts = NnetSimpleComputationOptions()
    asr = NnetLatticeFasterRecognizer.from_files(
        model_path, graph_path, symbols_path,
        decoder_opts=decoder_opts,
        decodable_opts=decodable_opts)

    # Read the features
    feature_reader = SequentialBaseFloatMatrixReader(features_rspecifier)

    # Process each utterance
    for key, feats in feature_reader:
        out = asr.decode(feats)
        print(key, out["text"])