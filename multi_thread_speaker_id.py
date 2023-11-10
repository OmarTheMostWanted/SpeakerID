import multiprocessing as mp
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import tqdm  # for the progress bar
import pickle
import concurrent.futures

def extract_features(filename):

    # Load the audio file
    audio, sample_rate = librosa.load(filename)

    print(f"Extracting MFCCs features from {filename}")
    # Extract MFCCs from the audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    print(f"Extracting Chroma features from {filename}")
    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)

    print(f"Extracting Spectral Contrast features from {filename}")
    # Extract Spectral Contrast features
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)

    print(f"Extracting Tonnetz features from {filename}")
    # Extract Tonnetz features
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    tonnetz_processed = np.mean(tonnetz.T, axis=0)

    print(f"Concatenating features from {filename}")
    # Concatenate all features into one array
    features = np.concatenate(
        [mfccs_processed, chroma_processed, spec_contrast_processed, tonnetz_processed]
    )
    return features

def extract_features_asyncronous(filename, threads=4):
    """
    This function extracts audio features from a given audio file using multiple threads.
    
    Parameters:
    filename (str): The path to the audio file.
    threads (int): The number of threads to use for feature extraction. Default is 4.

    Returns:
    np.array: A numpy array containing the extracted features.
    """

    # Load the audio file
    audio, sample_rate = librosa.load(filename)

    # Define the functions to be executed in separate threads

    def extract_mfccs():
        """
        MFCCs (Mel-frequency cepstral coefficients) provide a small set of features 
        which concisely describe the overall shape of a spectral envelope. 
        In MIR, it is often used to describe timbre.
        """
        print(f"Extracting MFCCs features from {filename}")
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)

    def extract_chroma():
        """
        Chroma features are an interesting and powerful representation for music audio 
        in which the entire spectrum is projected onto 12 bins representing the 12 
        distinct semitones (or chroma) of the musical octave. 
        """
        print(f"Extracting Chroma features from {filename}")
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        return np.mean(chroma.T, axis=0)

    def extract_spec_contrast():
        """
        Spectral contrast is defined as the difference in amplitude between peaks and valleys in a sound spectrum.
        It provides a measure of spectral shape that has been shown to be important in the perception of timbre.
        """
        print(f"Extracting Spectral Contrast features from {filename}")
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        return np.mean(spec_contrast.T, axis=0)

    def extract_tonnetz():
        """
        The tonnetz is a representation of musical pitch classes that has been used to analyze harmony in Western tonal music.
        It can be useful for key detection and chord recognition.
        """
        print(f"Extracting Tonnetz features from {filename}")
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
        return np.mean(tonnetz.T, axis=0)

    # Use a ThreadPoolExecutor to execute the functions in separate threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        
        # Submit all tasks to executor
        futures = []
        futures.append(executor.submit(extract_mfccs))
        futures.append(executor.submit(extract_chroma))
        futures.append(executor.submit(extract_spec_contrast))
        futures.append(executor.submit(extract_tonnetz))

        # Wait for all tasks to complete and retrieve their results
        results = [f.result() for f in futures]

    print(f"Concatenating features from {filename}")
    
    # Concatenate all features into one array
    features = np.concatenate(results)
    
    return features



def read_files(audio_files_dir, threads):

    # Create a pool of processes
    pool = mp.Pool(processes=threads)

    data = []
    labels = []
    for speaker in os.listdir(audio_files_dir):
        print(f"reading files for {speaker}")

        # Get a list of all the audio files for this speaker
        audio_files = [
            audio_files_dir + "/" + speaker + "/" + filename
            for filename in os.listdir(audio_files_dir + "/" + speaker)
            if filename.endswith(".wav")
        ]

        # Use the pool to extract features from all files in parallel and add a progress bar
        results = list(
            tqdm.tqdm(
                pool.imap(extract_features, audio_files),
                total=len(audio_files),
                desc=f"Processing {speaker}'s files",
            )
        )
        data.extend(results)
        labels.extend([speaker] * len(results))


    # Close the pool of processes
    pool.close()
    # Wait for all the threads to finish
    pool.join()

    return (data , labels)


def TrainSupportVectorClassification(data , labels):
    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Encode labels to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    # Define the model
    model = SVC(probability=True)

    # Train the model and add a progress message
    print("Training model...")
    model.fit(X_train, y_train)

    # Test the model and add a progress message
    print("Testing model...")
    y_pred = model.predict(X_test)

    # Calculate the accuracy of our model and print it out with more information
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    return (model , le)


def save_model(model , le):
    # Save the trained model and add a progress message
    print("Saving model...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save the label encoder and add a progress message
    print("Saving label encoder...")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

def load_model(model_file, label_encoder_file):
    # Load the trained model and add a progress message
    print("Loading model...")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load the label encoder and add a progress message
    print("Loading label encoder...")
    with open(label_encoder_file, "rb") as f:
        le = pickle.load(f)

    # Now you can use `model.predict_proba()` to get prediction probabilities for new data
    return (model , le)

def predict_speaker(model , le , threads=0):
    while True:
        # Ask user for an audio file path and identify it.
        audio_file_path = input("Please enter an audio file path or x to close: ")
        if audio_file_path == "x":
            quit(0)
        features = threads > 0 and extract_features_asyncronous(audio_file_path , threads) or  extract_features(audio_file_path)
        speaker_id = model.predict([features])
        speaker_name = le.inverse_transform(speaker_id)
        print(f"The speaker is: {speaker_name[0]}")
        return speaker_name

def predict_speaker_with_probability(model, le , threads=0):
    while True:
        # Ask user for an audio file path and identify it.
        audio_file_path = input("Please enter an audio file path or 'x' to close: ")
        if audio_file_path.lower() == "x":
            print("Closing the program.")
            break
        features = threads > 0 and extract_features_asyncronous(audio_file_path , threads) or extract_features(audio_file_path)
        speaker_id = model.predict([features])
        speaker_name = le.inverse_transform(speaker_id)
        
        # Get the probability of the prediction
        probability = model.predict_proba([features])
        max_prob_index = np.argmax(probability)
        max_prob = probability[0][max_prob_index]
        
        print(f"The speaker is: {speaker_name[0]} with a probability of {max_prob*100:.2f}%")

        return speaker_name, probability

if __name__ == "__main__":

    features = extract_features_asyncronous("audio_files_wav/Bryn Roberts/eventsof1848_01_milnes_64kb.wav" , 10)
    
    print(type(features))
    print(features)

    # data , labels = read_files("audio_files_wav", threads=8)

    # model , le = Train(data , labels)

    # save_model(model , le)

    # # model , le = load_model("model_91.67.pkl" , "label_encoder_91.67.pkl")

    # predict_speaker_with_probability(model , le)
