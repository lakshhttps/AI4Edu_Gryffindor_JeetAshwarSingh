import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

TEST_VIDEO_DIR = "../dataset/Test"
MODEL_PATH = "multiclass_model.pth"
IMG_SIZE = 224
SEQ_LEN = 16
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestDataset(Dataset):
    def _init_(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.videos = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith(('.mp4', '.avi', '.wmv', '.webm'))
        ])

    def _len_(self):
        return len(self.videos)

    def _getitem_(self, idx):
        vid_name = self.videos[idx]
        vid_path = os.path.join(self.video_dir, vid_name)
        frames = self._load_video(vid_path)

        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        else:
            to_tensor = transforms.ToTensor()
            frames = torch.stack([to_tensor(f) for f in frames])

        return frames, vid_name

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
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
            frames.append(frames[-1] if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        
        return frames[:SEQ_LEN]

class ResNetLSTM(nn.Module):
    def _init_(self):
        super(ResNetLSTM, self)._init_()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        feats = self.features(x).view(b, t, 512)
        lstm_out, _ = self.lstm(feats)
        return self.classifier(lstm_out[:, -1, :])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = ResNetLSTM().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = TestDataset(TEST_VIDEO_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

results = []

with torch.no_grad():
    for frames, names in loader:
        frames = frames.to(DEVICE)
        outputs = model(frames)
        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        
        for name, pred in zip(names, preds):
            results.append({"video": name, "label": pred[0]})

results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
print("Inference completed. Results saved to results.csv")