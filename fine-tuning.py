import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Config
TARGET_SAMPLE_RATE = 27000
FIXED_LENGTH = round(TARGET_SAMPLE_RATE * 5)  # 5 seconds

class Audio(Dataset):
    def __init__(self, file_paths, class_ids):
        self.file_paths = file_paths
        self.class_ids = class_ids
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE, n_mels=64
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            waveform, sr = torchaudio.load(path, normalize=True)

            if sr != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)

            audio_mono = torch.mean(waveform, dim=0, keepdim=True)

            if audio_mono.shape[1] < FIXED_LENGTH:
                pad_size = FIXED_LENGTH - audio_mono.shape[1]
                audio_mono = torch.nn.functional.pad(audio_mono, (0, pad_size))
            else:
                audio_mono = audio_mono[:, :FIXED_LENGTH]

            mel_specgram = self.mel_transform(audio_mono)
            mel_specgram = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()

            return {
                "specgram": mel_specgram.squeeze(0),
                "label": torch.tensor(self.class_ids[idx], dtype=torch.float32)
            }
        except Exception:
            return self.__getitem__((idx + 1) % len(self.file_paths))

class AudioWaveRecognizer(nn.Module):
    def __init__(self):
        super(AudioWaveRecognizer, self).__init__()
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

# Učitavanje pozitivnih podataka za fine-tuning
fine_tune_fart_folder = "/home/predrag/PycharmProjects/Leetcode/voiceGamesAI/farts_live2"
fine_tune_fart_files = [os.path.join(fine_tune_fart_folder, f) for f in os.listdir(fine_tune_fart_folder)]
fine_tune_fart_labels = [1.0] * len(fine_tune_fart_files)

# Dataset i DataLoader
fine_tune_dataset = Audio(fine_tune_fart_files, fine_tune_fart_labels)
fine_tune_dataloader = DataLoader(fine_tune_dataset, batch_size=16, shuffle=True)

# Učitavanje prethodno treniranog modela
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioWaveRecognizer().to(device)
model.load_state_dict(torch.load("/home/predrag/PycharmProjects/Leetcode/voiceGamesAI/S10.pth", map_location=device))
print("Učitani prethodno trenirani parametri iz S10.pth")

# Fine-tuning funkcija
def fine_tune(model, dataloader, device, epochs=20):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Fine-Tune Epoch {epoch + 1}"):
            inputs = batch['specgram'].unsqueeze(1).to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Fine-tune Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"fine_tuned_epoch{epoch + 1}.pth")
    print(f"Model saved as => fine_tuned_epoch{epoch + 1}.pth")

print(f"Broj novih pozitivnih uzoraka: {len(fine_tune_fart_files)}")
fine_tune(model, fine_tune_dataloader, device)
