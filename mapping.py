import numpy as np
from scipy.stats import entropy

# # color_map = [
# #     (0, "deep red"), (10, "orange"), (20, "yellow"), (30, "green"),
# #     (40, "cyan"), (50, "blue"), (60, "indigo"), (70, "violet")
# # ]

# # atmosphere_map = [
# #     (0, "dense fog"), (20, "misty dawn"), (40, "clear midday"),
# #     (60, "vivid sunset"), (80, "electric storm")
# # ]

# # emotion_map = [
# #     (0, "rested"), (20, "calmed"), (40, "focused"),
# #     (60, "stimulated"), (80, "agitated")
# # ]

# # element_map = [
# #     (0, "carbon"), (10, "iron"), (20, "copper"),
# #     (30, "silver"), (40, "gold"), (100, "neon")
# # ]

# def map_value(value, mapping):
#     keys = [k for k, _ in mapping]
#     texts = [v for _, v in mapping]
#     idx = np.searchsorted(keys, value, side="right") - 1
#     return texts[max(0, min(idx, len(texts)-1))]

# def get_entropy(Sxx):
#     P = np.abs(Sxx).flatten()
#     P = P / np.sum(P) if np.sum(P) > 0 else np.ones_like(P) / len(P)
#     P = np.clip(P, 1e-12, 1)  # avoid log(0)
#     return entropy(P, base=2)  

# def spectrogram_to_prompt(Sxx, fs=250):
#     if np.min(Sxx) < 0:
#         Sxx_lin = 10 ** (Sxx / 10.0)  # convert dB -> linear
#     else:
#         Sxx_lin = Sxx.copy()

#     # Normalize power matrix
#     Sxx_norm = Sxx_lin / np.max(Sxx_lin) if np.max(Sxx_lin) > 0 else Sxx_lin

#     # Extract features
#     mean_power = np.mean(Sxx_norm)
#     spectral_entropy = get_entropy(Sxx_norm)
#     freq_marginal = np.mean(Sxx_norm, axis=1)
#     centroid = np.sum(np.arange(len(freq_marginal)) * freq_marginal) / np.sum(freq_marginal)

#     # Normalize
#     centroid_norm = np.clip(100 * centroid / len(freq_marginal), 0, 100)
#     entropy_norm = np.clip(100 * spectral_entropy / np.log2(len(Sxx_norm.flatten())), 0, 100)
#     power_norm = np.clip(100 * (mean_power / np.max(Sxx_norm)), 0, 100)

#     # Mapping
#     color_map = [
#         (0, "deep red"), (15, "orange"), (30, "yellow"), (45, "green"),
#         (60, "cyan"), (75, "blue"), (90, "violet")
#     ]
#     atmosphere_map = [
#         (0, "murky fog"), (20, "hazy dawn"), (40, "clear afternoon"),
#         (60, "glowing sunset"), (80, "electric storm")
#     ]
#     emotion_map = [
#         (0, "rested"), (20, "calmed"), (40, "focused"),
#         (60, "stimulated"), (80, "agitated")
#     ]
#     element_map = [
#         (0, "carbon"), (20, "iron"), (40, "copper"),
#         (60, "silver"), (80, "gold"), (100, "neon")
#     ]

#     def map_value(value, mapping):
#         keys = [k for k, _ in mapping]
#         texts = [v for _, v in mapping]
#         idx = np.searchsorted(keys, value, side="right") - 1
#         return texts[max(0, min(idx, len(texts)-1))]

#     color = map_value(centroid_norm, color_map)
#     atmosphere = map_value(power_norm, atmosphere_map)
#     emotion = map_value(entropy_norm, emotion_map)
#     element = map_value((centroid_norm + power_norm) / 2, element_map)

#     prompt = (
#         f"An abstract composition inspired by EEG patterns, "
#         f"evoking a {emotion} emotional tone. "
#         f"The atmosphere feels like {atmosphere}, "
#         f"dominated by hues of {color}. "
#         f"It carries the elemental quality of {element}, "
#         f"expressing inner mental dynamics as visual form."
#     )

#     return {
#         "color": color,
#         "atmosphere": atmosphere,
#         "emotion": emotion,
#         "element": element,
#         "centroid_norm": centroid_norm,
#         "entropy_norm": entropy_norm,
#         "power_norm": power_norm,
#         "prompt": prompt
#     }

# if __name__ == "__main__":
#     Sxx = np.load("spectrograms/s01_spectrogram.npy")

#     mapped = spectrogram_to_prompt(Sxx)
#     print(mapped)

import os
import json
import numpy as np
from scipy.stats import entropy

# --- Helper functions ---
def safe_entropy(Sxx):
    P = np.abs(Sxx).flatten()
    P = P / np.sum(P) if np.sum(P) > 0 else np.ones_like(P) / len(P)
    P = np.clip(P, 1e-12, 1)
    return entropy(P, base=2)

def extract_features(Sxx):
    """Compute normalized features from a spectrogram."""
    if np.min(Sxx) < 0:  # Convert dB to linear if needed
        Sxx_lin = 10 ** (Sxx / 10.0)
    else:
        Sxx_lin = Sxx.copy()
    Sxx_norm = Sxx_lin / np.max(Sxx_lin) if np.max(Sxx_lin) > 0 else Sxx_lin
    mean_power = np.mean(Sxx_norm)
    spectral_entropy = safe_entropy(Sxx_norm)
    freq_marginal = np.mean(Sxx_norm, axis=1)
    centroid = np.sum(np.arange(len(freq_marginal)) * freq_marginal) / np.sum(freq_marginal)

    centroid_norm = centroid / len(freq_marginal)
    entropy_norm = spectral_entropy / np.log2(len(Sxx_norm.flatten()))
    power_norm = mean_power / np.max(Sxx_norm)
    return centroid_norm, power_norm, entropy_norm


def compute_percentile_bins(values, n_bins=5):
    """Compute percentile-based bins for dynamic mapping."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    return np.percentile(values, percentiles)


def map_dynamic(value, bins, labels):
    idx = np.searchsorted(bins, value, side="right") - 1
    return labels[max(0, min(idx, len(labels)-1))]


def adaptive_prompt(Sxx, feature_bins):
    """Map spectrogram features to descriptive prompt using adaptive thresholds."""
    centroid_norm, power_norm, entropy_norm = extract_features(Sxx)

    color_labels = ["deep red", "orange", "yellow", "green", "blue", "violet"]
    atmosphere_labels = ["murky fog", "misty morning", "clear sky", "sunset glow", "electric storm"]
    emotion_labels = ["rested", "calmed", "focused", "stimulated", "agitated"]
    element_labels = ["carbon", "iron", "copper", "silver", "gold", "neon"]

    color = map_dynamic(centroid_norm, feature_bins["centroid"], color_labels)
    atmosphere = map_dynamic(power_norm, feature_bins["power"], atmosphere_labels)
    emotion = map_dynamic(entropy_norm, feature_bins["entropy"], emotion_labels)
    element = map_dynamic((centroid_norm + power_norm) / 2, feature_bins["centroid"], element_labels)

    prompt = (
        f"An abstract composition inspired by EEG patterns, "
        f"evoking a {emotion} emotional tone. "
        f"The atmosphere feels like {atmosphere}, "
        f"dominated by hues of {color}. "
        f"It carries the elemental quality of {element}, "
        f"expressing inner mental dynamics as visual form."
    )

    return {
        "color": color,
        "atmosphere": atmosphere,
        "emotion": emotion,
        "element": element,
        "centroid_norm": centroid_norm,
        "entropy_norm": entropy_norm,
        "power_norm": power_norm,
        "prompt": prompt
    }


def process_all_spectrograms(folder="spectrograms", out_file="eeg_prompts.json"):
    """Compute adaptive prompts for all spectrograms in a folder."""
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if not files:
        print("No spectrogram files found.")
        return

    # collect feature statistics
    feats = [extract_features(np.load(os.path.join(folder, f))) for f in files]
    feats = np.array(feats)
    centroid_vals, power_vals, entropy_vals = feats[:,0], feats[:,1], feats[:,2]

    # compute adaptive bins
    feature_bins = {
        "centroid": compute_percentile_bins(centroid_vals, 5),
        "power": compute_percentile_bins(power_vals, 5),
        "entropy": compute_percentile_bins(entropy_vals, 5)
    }

    # generate prompts
    results = {}
    for fname in files:
        Sxx = np.load(os.path.join(folder, fname))
        results[fname] = adaptive_prompt(Sxx, feature_bins)

    # save results
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved adaptive prompts for {len(files)} spectrograms -> {out_file}")

    return results, feature_bins

if __name__ == "__main__":
    results, bins = process_all_spectrograms("spectrograms")