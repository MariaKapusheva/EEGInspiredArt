import numpy as np
from scipy.signal import argrelmax, argrelmin, welch
from scipy.interpolate import CubicSpline
import os
import glob

# EMD Functions
def find_extrema(signal):
    max_idx = argrelmax(signal)[0]
    min_idx = argrelmin(signal)[0]
    return max_idx, min_idx

def get_envelopes(signal, t, max_idx, min_idx):
    upper = CubicSpline(max_idx, signal[max_idx], bc_type='natural')(t)
    lower = CubicSpline(min_idx, signal[min_idx], bc_type='natural')(t)
    return upper, lower

def sift(signal, t, sift_thresh=0.05, max_sifts=10):
    h = signal.copy()
    for i in range(max_sifts):
        max_idx, min_idx = find_extrema(h)
        if len(max_idx) < 2 or len(min_idx) < 2:
            break
        upper, lower = get_envelopes(h, t, max_idx, min_idx)
        mean_env = (upper + lower) / 2
        h -= mean_env
        sd = np.sum(mean_env**2) / np.sum(h**2)
        if sd < sift_thresh:
            break
    return h

def emd(signal, max_imf=-1, sift_thresh=0.05, max_sifts=10):
    t = np.arange(len(signal))
    residue = signal.copy()
    imfs = []
    while True:
        imf = sift(residue, t, sift_thresh, max_sifts)
        residue -= imf
        imfs.append(imf)
        max_idx, min_idx = find_extrema(residue)
        if len(max_idx) < 2 or len(min_idx) < 2:
            break
        if max_imf > 0 and len(imfs) >= max_imf:
            break
    imfs.append(residue)
    return np.stack(imfs)

# Lempel-Ziv Complexity
def complexity_symbolize(signal, method='mean'):
    if method == 'mean':
        return (signal > np.mean(signal)).astype(int)
    else:
        raise ValueError("Unsupported symbolize method")

def _complexity_lempelziv_count(symbolic):
    string = "".join(list(symbolic.astype(str)))
    n = len(string)
    s = "0" + string
    c = 1
    j = 1
    i = 0
    k = 1
    k_max = 1
    stop = False
    while not stop:
        if s[i + k] != s[j + k]:
            if k > k_max:
                k_max = k
            i += 1
            if i == j:
                c += 1
                j += k_max
                if j + 1 > n:
                    stop = True
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
        else:
            k += 1
            if j + k > n:
                c += 1
                stop = True
    return c, n

def complexity_lempelziv(signal, normalize=True):
    symbolic = complexity_symbolize(signal, method='mean')
    c, n = _complexity_lempelziv_count(symbolic)
    if normalize:
        lzc = (c * np.log2(n)) / n
    else:
        lzc = c
    return lzc

# Main processing function
def process_eeg_buffer(buffer, fs, buffer_duration=4):
    # Extract peak frequencies using EMD + Welch
    imfs = emd(buffer)
    peaks = []
    for imf in imfs[:-1]:  # Exclude residue
        f, Pxx = welch(imf, fs=fs, nperseg=min(256, len(imf)))
        peaks.append(f[Pxx.argmax()])

    # Ratios between peaks
    ratios = []
    for i in range(len(peaks)):
        for j in range(i+1, len(peaks)):
            if peaks[j] != 0:
                ratios.append(peaks[i] / peaks[j])

    # Average power in frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'low-beta': (12, 20),
        'high-beta': (20, 30),
        'gamma': (30, 45)
    }
    f, Pxx = welch(buffer, fs=fs, nperseg=min(256, len(buffer)))
    powers = {}
    for name, (low, high) in bands.items():
        idx = (f >= low) & (f < high)
        powers[name] = np.mean(Pxx[idx]) if np.any(idx) else 0

    # Lempel-Ziv complexity
    lzc = complexity_lempelziv(buffer)

    return {
        'peak_frequencies': peaks,
        'peak_ratios': ratios,
        'band_powers': powers,
        'lzc': lzc
    }

# def process_dataset(directory, fs=250, file_pattern='s*.csv', buffer_duration=4, out_dir='features'):
#     import os, json
#     os.makedirs(out_dir, exist_ok=True)

#     files = glob.glob(os.path.join(directory, file_pattern))
#     if not files:
#         print("No files found! Check your directory or file pattern.")
#         return

#     for file in files:
#         try:
#             eeg_data = np.loadtxt(file, delimiter=',').flatten()
#         except Exception as e:
#             print(f"Failed to load {file}: {e}")
#             continue

#         subject_id = os.path.basename(file).split('.')[0]
#         buffer_size = int(buffer_duration * fs)
#         features = []

#         for start in range(0, len(eeg_data), buffer_size):
#             buffer = eeg_data[start:start + buffer_size]
#             if len(buffer) < buffer_size:
#                 continue
#             feat = process_eeg_buffer(buffer, fs, buffer_duration)
#             features.append(feat)

#         # Save one JSON file per EEG file
#         out_file = os.path.join(out_dir, f"{subject_id}_features.json")
#         with open(out_file, 'w') as f:
#             json.dump(features, f, indent=4)
#         print(f"Saved features for {subject_id} -> {out_file}")

def process_dataset(directory, fs=250, file_pattern='s*.csv', out_dir='features'):
    import os, json
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(directory, file_pattern))
    if not files:
        print("No files found! Check your directory or file pattern.")
        return

    results = {}

    for file in files:
        try:
            eeg_data = np.loadtxt(file, delimiter=',').flatten()
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            continue

        subject_id = os.path.basename(file).split('.')[0]

        # Process the entire EEG file as one signal
        feat = process_eeg_buffer(eeg_data, fs)
        results[subject_id] = feat

        # Save one JSON file per EEG file
        out_file = os.path.join(out_dir, f"{subject_id}_features.json")
        with open(out_file, 'w') as f:
            json.dump(feat, f, indent=4)
        print(f"Saved features for {subject_id} -> {out_file}")

    return results

results = process_dataset('archive', fs=250)

# Save all results together too
import json
with open('eeg_features.json', 'w') as f:
    json.dump(results, f, indent=4)
# results = process_dataset('archive', fs=250, buffer_duration=4)
# import json

# with open('eeg_features.json', 'w') as f:
#     json.dump(results, f, indent=4)
# # TODO: instead of segmenting every 4 seconds, do the extraction per entire file