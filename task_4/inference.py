import os
import numpy as np
import joblib
from scipy.signal import butter, filtfilt, welch, find_peaks


FS = 30
MODEL_PATH = "task_4_model.pkl"
SCALER_PATH = "task_4_scaler.pkl"
TEST_DATA_PATH = "dataset/Test"   

try:
    trapz = np.trapz
except AttributeError:
    trapz = np.trapezoid


def bandpass(signal, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_rppg_features(rppg, fs=30):
    if len(rppg) < fs:
        return np.zeros(12)

    rppg = bandpass(rppg, fs)
    rppg = (rppg - np.mean(rppg)) / (np.std(rppg) + 1e-6)

    peaks, _ = find_peaks(rppg, distance=int(fs * 0.5))

    if len(peaks) < 2:
        rr_intervals = np.array([1])
    else:
        rr_intervals = np.diff(peaks) / fs

    hr = 60 / (np.mean(rr_intervals) + 1e-6)
    hrv_sdnn = np.std(rr_intervals)

    signal_power = np.var(rppg)
    noise_power = np.var(np.diff(rppg)) + 1e-6
    sqi = signal_power / noise_power

    freqs, psd = welch(rppg, fs=fs, nperseg=min(256, len(rppg)))
    total_power = trapz(psd, freqs)

    return np.array([
        hr,
        hrv_sdnn,
        sqi,
        total_power,
        np.mean(rppg),
        np.std(rppg),
        np.max(rppg),
        np.min(rppg),
        len(peaks),
        np.percentile(rppg, 25),
        np.percentile(rppg, 50),
        np.percentile(rppg, 75)
    ], dtype=np.float32)

print("[INFO] Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("[INFO] Model loaded successfully")

results = []

for file in sorted(os.listdir(TEST_DATA_PATH)):
    if not file.endswith(".npy"):
        continue

    file_path = os.path.join(TEST_DATA_PATH, file)
    rppg_signal = np.load(file_path)

    features = extract_rppg_features(rppg_signal, FS)
    features = np.nan_to_num(features).reshape(1, -1)

    features_scaled = scaler.transform(features)

    pred_class = model.predict(features_scaled)[0]
    pred_prob = model.predict_proba(features_scaled)[0]

    results.append((file, pred_class, pred_prob))

    print(f"[PRED] {file}")
    print(f"       Class      : {pred_class}")
    print(f"       Probabilities: {pred_prob}")

print("\n[INFO] Inference completed on all test files.")