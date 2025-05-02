import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
import os
import soundfile as sf
from datetime import datetime


# Configurations 
TARGET_SAMPLE_RATE = 22050
RECORD_SECONDS = 1
SOUND_LEN = TARGET_SAMPLE_RATE * RECORD_SECONDS
N_MELS = 64 #standard number of mels

class ReduceSumLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)


# Normalize audio to fit for the model
def audio_normalizer(sound):
    if len(sound) < SOUND_LEN: # if len is less than predefined
        add_len = SOUND_LEN - len(audio) # increse for add_len
        sound = np.pad(sound, (0, add_len))# add zeros
    else:
        sound = audio[:SOUND_LEN] # Else, just cut 

    spectrogram = librosa.feature.melspectrogram(y=sound, sr=TARGET_SAMPLE_RATE, n_mels=N_MELS) #Generate spectrogram
    spec_db = librosa.power_to_db(spectrogram, ref=np.max) #Generate spectrogram to decibels
    result = (spec_db - np.mean(spec_db)) / np.std(spec_db) # Z-score normalize

    return result.astype(np.float32)


# Laod model
model = tf.keras.models.load_model("Models/Model_V3.keras",custom_objects={"ReduceSumLayer": ReduceSumLayer})


#Main part
while True:
    print("Listening...")
    audio = sd.rec(int(RECORD_SECONDS * TARGET_SAMPLE_RATE), samplerate=TARGET_SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()

    audio = audio.flatten() # Remove additional dimensions

    rms = np.sqrt(np.mean(audio**2))
    db = 20 * np.log10(rms + 1e-10)# convert to decibels
    if db < -20:# check loudness
        print(f"YOU NEED TO BE MORE LOUD!")
        continue
    '''if db>-1:#MAYBE WE DONT NEED TO CHECK IF SOUND IS TOO LOUD
        print(f"YOU ARE TO LOUD!!!")
        continue'''



    mel = audio_normalizer(audio)

    mel = mel[np.newaxis, ..., np.newaxis]  #batch (first 1) and channels (the last 1)
    audio_arr = model.predict(mel)[0][0]
    p = tf.sigmoid(audio_arr).numpy()

    if p >= 0.7:
        print(f"FART {p}")
    else:
        print(f"UNKNOWN {p}")