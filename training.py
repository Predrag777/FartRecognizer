import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# Config
TARGET_SAMPLE_RATE = 27000### 29 aprila u 9:04 povecao sa 2 sekunde na 5. Linija 12!!!
FIXED_LENGTH = round(TARGET_SAMPLE_RATE * 5)  # 2 seconds


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


# Paths
fart_folder = "/home/predrag/PycharmProjects/Leetcode/FartsV2"
fart_folder2 = "/home/predrag/PycharmProjects/Leetcode/audio_dataset/train/Fart"

car_horn_folder = "/home/predrag/PycharmProjects/Leetcode/TrimmedSounds/car_horn_trimmed"
dog_barking_folder = "/home/predrag/PycharmProjects/Leetcode/TrimmedSounds/dog_barking_trimmed"
guitar_folder = "/home/predrag/PycharmProjects/Leetcode/TrimmedSounds/guitar_trimmed"
knock_folder = "/home/predrag/PycharmProjects/Leetcode/TrimmedSounds/Knock_trimmed"
laughter_folder = "/home/predrag/PycharmProjects/Leetcode/TrimmedSounds/laughter_trimmed"
gunshot_folder = "/home/predrag/PycharmProjects/Leetcode/TrimmedSounds/gunshot_trimmed"
siren_folder = "/home/predrag/PycharmProjects/Leetcode/siren_trimmed"
snare_folder = "/home/predrag/PycharmProjects/Leetcode/TrimmedSounds/Snare_drum_trimmed"
human_folder = "/home/predrag/PycharmProjects/Leetcode/Human_voice"

# Dataset loading: Combine both fart folders
fart_files = []

# Load files from both folders
for folder in [fart_folder, fart_folder2]:
    fart_files += [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith('.wav')
    ]

# Shuffle combined list
random.shuffle(fart_files)

other_files = []
for folder in [laughter_folder,human_folder]:
    other_files.extend([os.path.join(folder, f) for f in os.listdir(folder)])

files = fart_files + other_files
labels = [1.0] * len(fart_files) + [0.0] * len(other_files)

combined = list(zip(files, labels))
random.shuffle(combined)
files, labels = zip(*combined)
files = list(files)
labels = list(labels)

# Split
split_idx = int(0.8 * len(files))
train_files, val_files = files[:split_idx], files[split_idx:]
train_labels, val_labels = labels[:split_idx], labels[split_idx:]

train_dataset = Audio(train_files, train_labels)
val_dataset = Audio(val_files, val_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Test set
test_fart_folder = "/home/predrag/PycharmProjects/Leetcode/TrimmedSounds/fart_trimmed"
test_fart_files = [os.path.join(test_fart_folder, f) for f in os.listdir(test_fart_folder)]
test_fart_labels = [1.0] * len(test_fart_files)

negative_test_folders = [
    "/home/predrag/PycharmProjects/Leetcode/Human_voice"
]

negative_test_files = []
for folder in negative_test_folders:
    negative_test_files.extend([os.path.join(folder, f) for f in os.listdir(folder)])
negative_test_labels = [0.0] * len(negative_test_files)

test_files = test_fart_files + negative_test_files
test_labels = test_fart_labels + negative_test_labels

combined_test = list(zip(test_files, test_labels))
random.shuffle(combined_test)
test_files, test_labels = zip(*combined_test)
test_files = list(test_files)
test_labels = list(test_labels)

test_dataset = Audio(test_files, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['specgram'].unsqueeze(1).to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions.float() == labels).sum().item()
            total += labels.size(0)
    return correct / total


def compute_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['specgram'].unsqueeze(1).to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(model, train_dataloader, val_dataloader, test_dataloader, device, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    pos_weight = torch.tensor([1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            inputs = batch['specgram'].unsqueeze(1).to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_dataloader)
        val_loss = compute_loss(model, val_dataloader, criterion, device)
        test_acc = evaluate(model, test_dataloader, device)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")
        torch.save(model.state_dict(), f"Model_V5.{epoch + 1}.pth")
        print(f"Model saved as => Model_V5.{epoch + 1}.pth")


# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioWaveRecognizer().to(device)

print(f"Farts: {len(fart_files)}    Others: {len(other_files)}")
print(f"Training samples: {len(train_files)}, Validation: {len(val_files)}, Test samples: {len(test_files)}")

train(model, train_dataloader, val_dataloader, test_dataloader, device)
