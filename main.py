import os
import librosa
from IPython.display import Audio
from collections import Counter
import numpy as np

# Define dataset paths
jazz = "/kaggle/input/jazz-vs-classical-music-classification/dataset/Jazz"
classical = "/kaggle/input/jazz-vs-classical-music-classification/dataset/Classical"

# Function to play the first five audio files from the dataset
def play_first_five_songs(genre, genre_path):
    """
    Play the first five songs from a given genre folder.
    """
    files_in_folder = os.listdir(genre_path)
    wav_files = [f for f in files_in_folder if f.endswith('.wav')][:5]

    if wav_files:
        for idx, file_name in enumerate(wav_files):
            file_path = os.path.join(genre_path, file_name)
            print(f"Playing {file_name} from {genre} genre")
            audio, sample_rate = librosa.load(file_path, sr=None)
            display(Audio(audio, rate=sample_rate))
    else:
        print(f"No .wav files found in {genre_path}")

# Play the first five songs from both Jazz and Classical genres
print("Playing first five Jazz songs:")
play_first_five_songs("Jazz", jazz)

print("\nPlaying first five Classical songs:")
play_first_five_songs("Classical", classical)

# Function to summarize audio durations
def summarize_audio_durations(directory):
    """
    Summarizes the durations of audio files in the specified directory.
    """
    audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    durations = []

    for audio_file in audio_files:
        audio_path = os.path.join(directory, audio_file)
        audio, sample_rate = librosa.load(audio_path)
        duration = librosa.get_duration(y=audio, sr=sample_rate)
        durations.append(duration)

    duration_counts = Counter(durations)

    print(f"Summary of Audio Durations (in seconds) for {directory}:")
    for duration, count in sorted(duration_counts.items()):
        print(f"Duration: {duration:.2f} seconds, Count: {count} files")
    print(f"Total audio files: {len(audio_files)}\n")

# Summarize the durations for both Jazz and Classical directories
summarize_audio_durations(jazz)
summarize_audio_durations(classical)


def delete_small_files(directory, target_duration=5.0):
    """
    Delete audio files with durations other than the specified target duration (default 5 seconds).
    """
    audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    for audio_file in audio_files:
        audio_path = os.path.join(directory, audio_file)
        try:
            audio, sample_rate = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sample_rate)

            if duration != target_duration:
                print(f"Deleting {audio_file} with duration {duration:.2f} seconds")
                os.remove(audio_path)
            else:
                print(f"Keeping {audio_file} with duration {duration:.2f} seconds")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

# Delete small files in both Jazz and Classical directories
delete_small_files(jazz)
delete_small_files(classical)


def check_summary_after_deletion(directory, target_duration=5.0):
    """
    Check and summarize the durations of audio files in the directory after deletion.
    """
    audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    durations = []

    for audio_file in audio_files:
        audio_path = os.path.join(directory, audio_file)
        audio, sample_rate = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=audio, sr=sample_rate)

        if duration == target_duration:
            durations.append(duration)

    duration_counts = Counter(durations)
    print(f"Summary of Audio Durations (in seconds) after deletion in {directory}:")
    for duration, count in sorted(duration_counts.items()):
        print(f"Duration: {duration:.2f} seconds, Count: {count} files")
    print(f"Total audio files after deletion: {len(durations)}")

# Check summaries for Jazz and Classical directories
check_summary_after_deletion(jazz)
check_summary_after_deletion(classical)


def summarize_sample_rates(directory):
    """
    Summarize the sample rates of audio files in the specified directory.
    """
    audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    sample_rates = []

    for audio_file in audio_files:
        audio_path = os.path.join(directory, audio_file)
        _, sample_rate = librosa.load(audio_path, sr=None)
        sample_rates.append(sample_rate)

    sample_rate_counts = Counter(sample_rates)
    print(f"Summary of Sample Rates for {directory}:")
    for sample_rate, count in sorted(sample_rate_counts.items()):
        print(f"Sample Rate: {sample_rate}, Count: {count} files")
    print(f"Total audio files: {len(audio_files)}")

# Summarize sample rates for Jazz and Classical directories
summarize_sample_rates(jazz)
summarize_sample_rates(classical)


def extract_mfcc(audio_path, n_mfcc=13):
    """
    Extract MFCC features from an audio file.
    """
    audio, sample_rate = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs.T  # Transposed to have time frames as rows

def extract_mel_spectrogram(audio_path):
    """
    Extract mel spectrogram from an audio file.
    """
    audio, sample_rate = librosa.load(audio_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    return mel_spectrogram

def augment_audio(audio, sample_rate):
    """
    Augment audio by adding noise.
    """
    noise = np.random.normal(0, 0.01, audio.shape)
    audio_noisy = audio + noise
    return audio_noisy


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

def load_audio_files_and_labels(jazz_path, classical_path):
    """
    Load audio file paths and their corresponding labels.

    Parameters:
        jazz_path (str): Path to the Jazz audio files.
        classical_path (str): Path to the Classical audio files.

    Returns:
        audio_files (list): List of audio file paths.
        labels (list): Corresponding labels for the audio files.
    """
    audio_files = []
    labels = []

    # Load Jazz files
    jazz_files = [os.path.join(jazz_path, f) for f in os.listdir(jazz_path) if f.endswith('.wav')]
    audio_files.extend(jazz_files)
    labels.extend(['Jazz'] * len(jazz_files))

    # Load Classical files
    classical_files = [os.path.join(classical_path, f) for f in os.listdir(classical_path) if f.endswith('.wav')]
    audio_files.extend(classical_files)
    labels.extend(['Classical'] * len(classical_files))

    return audio_files, labels

# Load data, extract features, and encode labels
audio_files, labels = load_audio_files_and_labels(jazz, classical)  
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Extract features from the audio files (MFCC in this case)
def extract_features(audio_files):
    features = []
    for audio_file in audio_files:
        audio, sample_rate = librosa.load(audio_file, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        features.append(np.mean(mfcc.T, axis=0))  # Average MFCCs across time frames
    return np.array(features)

# Extract features from the audio files
features = extract_features(audio_files)


from tensorflow.keras.utils import to_categorical

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Reshaping the data for RNN input: (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Reshaped to match RNN input requirements
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# One-hot encoding of labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense

# Define RNN model
input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
num_classes = y_train.shape[1]  # Number of classes (e.g., 2 for Jazz and Classical)

model = Sequential([
    SimpleRNN(64, input_shape=input_shape, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
print("Training the RNN model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))


# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


def classify_song(audio_path, model, label_encoder):
    """
    Classifies a song as Jazz or Classical based on its audio features.

    Parameters:
        audio_path (str): Path to the audio file.
        model (keras.Model): Trained RNN model.
        label_encoder (LabelEncoder): Encoder to map numerical labels to class names.

    Returns:
        str: Predicted genre (e.g., 'Jazz' or 'Classical').
    """
    import librosa
    import numpy as np
    
    # Load the audio file and extract features (MFCC)
    try:
        audio, sample_rate = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        feature = np.mean(mfcc.T, axis=0)  # Average MFCCs across time frames
        
        # Reshape the feature to match RNN input shape
        feature = feature.reshape(1, 1, -1)  # (samples, time_steps, features)

        # Predict the genre
        prediction = model.predict(feature)
        predicted_label = np.argmax(prediction, axis=1)  # Get the class index

        # Decode the label to its genre name
        predicted_genre = label_encoder.inverse_transform(predicted_label)[0]
        return predicted_genre
    except Exception as e:
        return f"Error processing the song: {e}"


# Example path to a new song
song_path = "/kaggle/input/jazz11/Fly Me To The Moon (2008 Remastered).mp3"

# Classify the song
predicted_genre = classify_song(song_path, model, label_encoder)
print(f"The song is classified as: {predicted_genre}")

# Example path to a new song
song_path = "/kaggle/input/classical/38218_beethoven_piano_sonata_no.14_opus_27_no.2_(moonlight_sonata_1st_movement)_proud_music_preview.mp3"

# Classify the song
predicted_genre = classify_song(song_path, model, label_encoder)
print(f"The song is classified as: {predicted_genre}")

