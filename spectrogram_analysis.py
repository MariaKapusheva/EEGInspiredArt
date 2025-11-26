import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_spectrograms(
    directory,
    fs=250,
    file_pattern='s*.csv',
    out_dir='spectrograms',
    save_images=True,
    save_arrays=True,
    nperseg=256,
    noverlap=128,
    cmap='magma'
):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(os.path.join(directory, file_pattern))
    
    if not files:
        print("No EEG files found.")
        return
    
    for file in files:
        try:
            eeg_data = np.loadtxt(file, delimiter=',').flatten()
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            continue
        
        subject_id = os.path.basename(file).split('.')[0]
        
        # Compute the spectrogram
        f, t, Sxx = spectrogram(eeg_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)  # Convert power to dB for better contrast
        
        # Save spectrogram array
        if save_arrays:
            np.save(os.path.join(out_dir, f"{subject_id}_spectrogram.npy"), Sxx_log)
        
        # Save image
        if save_images:
            plt.figure(figsize=(8, 4))
            plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap=cmap)
            plt.title(f"Spectrogram - {subject_id}")
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{subject_id}_spectrogram.png"))
            plt.close()
        
        print(f"Saved spectrogram for {subject_id}")
    
    print(f" All spectrograms are saved in '{out_dir}'")

generate_spectrograms('archive', fs=250)
