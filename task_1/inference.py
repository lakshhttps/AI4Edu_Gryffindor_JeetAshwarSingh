
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

TEST_VIDEO_FOLDER = "../data/Test"
MODEL_PATH = "best_binary_model.pth"
IMG_SIZE = 224
SEQ_LEN = 16
BATCH_SIZE = 4

class EngagementTestDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.videos = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith(('.mp4', '.avi', '.wmv', '.webm'))
        ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        vid_path = os.path.join(self.video_dir, vid_name)

        frames = self._load_video(vid_path)

        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        else:
            frames = torch.tensor(frames)

        return frames, vid_name

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)] * SEQ_LEN

        indices = np.linspace(0, total_frames - 1, SEQ_LEN).astype(int)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if len(frames) == SEQ_LEN:
                    break

        cap.release()

        while len(frames) < SEQ_LEN:
            frames.append(frames[-1])

        return frames[:SEQ_LEN]

class ResNetLSTM(nn.Module):
    def __init__(self):
        super(ResNetLSTM, self).__init__()
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        feats = self.features(x)
        feats = feats.view(batch_size, seq_len, 512)
        lstm_out, _ = self.lstm(feats)
        return self.classifier(lstm_out[:, -1, :])

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = ResNetLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

dataset = EngagementTestDataset(TEST_VIDEO_FOLDER, transform=tfms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

results = []

with torch.no_grad():
    for frames, names in loader:
        frames = frames.to(device)
        outputs = model(frames)
        probs = torch.sigmoid(outputs).squeeze(1)

        for name, p in zip(names, probs):
            results.append({
                "video": name,
                "probability": float(p),
                "prediction": int(p > 0.5)  # 0 = Low, 1 = High
            })

df = pd.DataFrame(results)
df.to_csv("test_predictions.csv", index=False)
print("Inference completed → test_predictions.csv")
