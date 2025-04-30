import torch
import torchaudio
import torch.nn as nn
import numpy as np
import sounddevice as sd
import librosa
import os
import time
import soundfile as sf

# ================== CONFIG ===================
TARGET_SAMPLE_RATE = 27000
RECORD_DURATION = 5  # seconds
AUDIO_LENGTH = TARGET_SAMPLE_RATE * RECORD_DURATION
FIXED_LENGTH = TARGET_SAMPLE_RATE * 5  # 5 seconds
THRESHOLD = 0.3
SAVE_DIR = "/home/predrag/PycharmProjects/Leetcode/voiceGamesAI/liveRecording2"

# ================== MODEL ====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.4),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.4),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = CNN()
model.load_state_dict(torch.load("/home/predrag/PycharmProjects/Leetcode/voiceGamesAI/models/model_29_04.pth", map_location=torch.device('cpu')))
model.eval()

# ============ SILENCE TRIMMING ===============
def remove_silence(audio_numpy, sample_rate):
    trimmed_audio, _ = librosa.effects.trim(audio_numpy, top_db=20)
    return trimmed_audio

# ============ AUDIO PREPROCESSING ============
def process(audio_numpy):
    audio_numpy = remove_silence(audio_numpy, TARGET_SAMPLE_RATE)

    if len(audio_numpy) < FIXED_LENGTH:
        pad_size = FIXED_LENGTH - len(audio_numpy)
        audio_numpy = np.pad(audio_numpy, (0, pad_size), mode='constant')
    else:
        audio_numpy = audio_numpy[:FIXED_LENGTH]

    tensor_wave = torch.tensor(audio_numpy, dtype=torch.float32).unsqueeze(0)  # [1, N]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_mels=64
    )
    mel_specgram = mel_transform(tensor_wave)
    mel_specgram = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()

    return mel_specgram.unsqueeze(0).to(torch.float32)  # [1, 1, 64, T]

# =============== PREDICTION ==================
def predict(audio_numpy, threshold=THRESHOLD):
    input_tensor = process(audio_numpy)
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
    return probability >= threshold, probability

# ============ SAVE TO FILE ====================
def save_recording(audio_numpy, directory):
    os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    path = os.path.join(directory, filename)
    sf.write(path, audio_numpy, TARGET_SAMPLE_RATE)
    return path

# =============== MAIN LOOP ===================
def main():
    print("Listening...\n")
    try:
        while True:
            recording = sd.rec(frames=AUDIO_LENGTH, samplerate=TARGET_SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            audio = np.squeeze(recording.T)

            # Save recording
            #save_path = save_recording(audio, SAVE_DIR)

            detected, probability = predict(audio)

            if detected:
                print(f"[FART DETECTED] ({probability:.2f}) => ")
            else:
                print(f"[UNKNOWN SOUND] ({probability:.2f}) =>")

    except KeyboardInterrupt:
        print("\nHALT: Stopped by user.")

if __name__ == "__main__":
    main()
