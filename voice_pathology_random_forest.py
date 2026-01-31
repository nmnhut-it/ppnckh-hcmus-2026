#!/usr/bin/env python3
"""
Voice Pathology Detection using Random Forest Classifier
=========================================================

Thesis: Building an Automated System for Pre- and Post-Surgical Voice Change
Assessment Based on Multidimensional Acoustic Feature Analysis
(F0, Formants, and Perturbation)

This script implements a Random Forest-based classification pipeline for
detecting voice pathology (Healthy vs. Pathological) using multidimensional
acoustic features extracted from sustained vowel recordings.

Supported Datasets:
    1. VOICED (PhysioNet)   - 208 samples, vowel /a/, 8 kHz
    2. SVD (Saarbrucken)    - 2000+ speakers, vowels /a,i,u/, 50 kHz
    3. AVFAD                - 709 subjects, vowels /a,e,o/, pre-extracted params
    4. FEMH                 - 2000 subjects, vowel /a/, 44.1 kHz

Required packages:
    pip install numpy pandas scikit-learn matplotlib seaborn parselmouth librosa scipy wfdb
"""

import os
import re
import json
import warnings
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. ACOUSTIC FEATURE EXTRACTION
# =============================================================================

class AcousticFeatureExtractor:
    """
    Extracts multidimensional acoustic features from voice recordings using
    Parselmouth (Praat interface) and Librosa.

    Feature groups:
        - Source features:       F0 (mean, std, min, max)
        - Filter features:       F1, F2, F3 (formant frequencies)
        - Perturbation features: Jitter, Shimmer, HNR, CPP
        - Spectral features:     MFCCs, spectral centroid, bandwidth, rolloff
    """

    def __init__(self, target_sr: int = 16000):
        """
        Args:
            target_sr: Target sample rate for resampling (default 16 kHz).
        """
        self.target_sr = target_sr

    def extract_all_features(self, signal: np.ndarray, sr: int) -> dict:
        """Extract all acoustic features from a voice signal.

        Args:
            signal: Audio signal as 1D numpy array.
            sr: Original sample rate.

        Returns:
            Dictionary of feature_name -> value.
        """
        import parselmouth
        from parselmouth.praat import call

        # Resample if needed
        if sr != self.target_sr:
            import librosa
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        # Normalize amplitude
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal))

        # Create Parselmouth Sound object
        snd = parselmouth.Sound(signal, sampling_frequency=sr)

        features = {}

        # --- Source features (F0) ---
        features.update(self._extract_f0_features(snd))

        # --- Filter features (Formants F1, F2, F3) ---
        features.update(self._extract_formant_features(snd))

        # --- Perturbation features (Jitter, Shimmer, HNR) ---
        features.update(self._extract_perturbation_features(snd))

        # --- Cepstral features (CPP) ---
        features.update(self._extract_cpp_features(snd))

        # --- Spectral features (MFCCs, spectral centroid, etc.) ---
        features.update(self._extract_spectral_features(signal, sr))

        return features

    def _extract_f0_features(self, snd) -> dict:
        """Extract fundamental frequency (F0) features."""
        from parselmouth.praat import call

        features = {}
        try:
            pitch = call(snd, "To Pitch", 0.0, 75.0, 600.0)
            f0_values = pitch.selected_array["frequency"]
            f0_voiced = f0_values[f0_values > 0]

            if len(f0_voiced) > 0:
                features["f0_mean"] = np.mean(f0_voiced)
                features["f0_std"] = np.std(f0_voiced)
                features["f0_min"] = np.min(f0_voiced)
                features["f0_max"] = np.max(f0_voiced)
                features["f0_range"] = np.max(f0_voiced) - np.min(f0_voiced)
                features["f0_median"] = np.median(f0_voiced)
            else:
                for key in ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_median"]:
                    features[key] = np.nan
        except Exception as e:
            logger.debug(f"F0 extraction failed: {e}")
            for key in ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_median"]:
                features[key] = np.nan

        return features

    def _extract_formant_features(self, snd) -> dict:
        """Extract formant frequencies (F1, F2, F3)."""
        from parselmouth.praat import call

        features = {}
        try:
            formants = call(snd, "To Formant (burg)", 0.0, 5, 5500.0, 0.025, 50.0)
            n_frames = call(formants, "Get number of frames")

            f1_vals, f2_vals, f3_vals = [], [], []
            for i in range(1, n_frames + 1):
                t = call(formants, "Get time from frame number", i)
                f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
                f3 = call(formants, "Get value at time", 3, t, "Hertz", "Linear")
                if not np.isnan(f1):
                    f1_vals.append(f1)
                if not np.isnan(f2):
                    f2_vals.append(f2)
                if not np.isnan(f3):
                    f3_vals.append(f3)

            for name, vals in [("f1", f1_vals), ("f2", f2_vals), ("f3", f3_vals)]:
                if vals:
                    features[f"{name}_mean"] = np.mean(vals)
                    features[f"{name}_std"] = np.std(vals)
                else:
                    features[f"{name}_mean"] = np.nan
                    features[f"{name}_std"] = np.nan
        except Exception as e:
            logger.debug(f"Formant extraction failed: {e}")
            for name in ["f1", "f2", "f3"]:
                features[f"{name}_mean"] = np.nan
                features[f"{name}_std"] = np.nan

        return features

    def _extract_perturbation_features(self, snd) -> dict:
        """Extract Jitter, Shimmer, and HNR."""
        from parselmouth.praat import call

        features = {}
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)

            # Jitter variants
            features["jitter_local"] = call(
                point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
            )
            features["jitter_local_abs"] = call(
                point_process, "Get jitter (local, absolute)", 0.0, 0.0, 0.0001, 0.02, 1.3
            )
            features["jitter_rap"] = call(
                point_process, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3
            )
            features["jitter_ppq5"] = call(
                point_process, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3
            )

            # Shimmer variants
            features["shimmer_local"] = call(
                [snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
            )
            features["shimmer_local_db"] = call(
                [snd, point_process], "Get shimmer (local_dB)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
            )
            features["shimmer_apq3"] = call(
                [snd, point_process], "Get shimmer (apq3)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
            )
            features["shimmer_apq5"] = call(
                [snd, point_process], "Get shimmer (apq5)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
            )

            # HNR
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
            features["hnr_mean"] = call(harmonicity, "Get mean", 0.0, 0.0)
            features["hnr_std"] = call(harmonicity, "Get standard deviation", 0.0, 0.0)

        except Exception as e:
            logger.debug(f"Perturbation extraction failed: {e}")
            for key in [
                "jitter_local", "jitter_local_abs", "jitter_rap", "jitter_ppq5",
                "shimmer_local", "shimmer_local_db", "shimmer_apq3", "shimmer_apq5",
                "hnr_mean", "hnr_std",
            ]:
                features[key] = np.nan

        return features

    def _extract_cpp_features(self, snd) -> dict:
        """Extract Cepstral Peak Prominence (CPP/CPPS)."""
        from parselmouth.praat import call

        features = {}
        try:
            # Compute power cepstrogram and get CPPS
            power_cepstrogram = call(snd, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50)
            cpps = call(
                power_cepstrogram, "Get CPPS",
                False,  # subtract tilt before smoothing
                0.02,   # time averaging window (s)
                0.0005, # quefrency averaging window (s)
                60.0,   # floor pitch (Hz)
                330.0,  # ceiling pitch (Hz)
                0.05,   # tolerance
                "Parabolic",    # interpolation
                "Exponential decay",  # tilt line quefrency range
                "Robust slow",  # fit method
            )
            features["cpps"] = cpps
        except Exception as e:
            logger.debug(f"CPP extraction failed: {e}")
            features["cpps"] = np.nan

        return features

    def _extract_spectral_features(self, signal: np.ndarray, sr: int) -> dict:
        """Extract MFCCs and other spectral features using Librosa."""
        import librosa

        features = {}
        try:
            # MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])
                features[f"mfcc_{i+1}_std"] = np.std(mfccs[i])

            # Delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs)
            for i in range(13):
                features[f"delta_mfcc_{i+1}_mean"] = np.mean(delta_mfccs[i])

            # Spectral centroid
            spec_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
            features["spectral_centroid_mean"] = np.mean(spec_centroid)
            features["spectral_centroid_std"] = np.std(spec_centroid)

            # Spectral bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
            features["spectral_bandwidth_mean"] = np.mean(spec_bw)

            # Spectral rolloff
            spec_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
            features["spectral_rolloff_mean"] = np.mean(spec_rolloff)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(signal)
            features["zcr_mean"] = np.mean(zcr)
            features["zcr_std"] = np.std(zcr)

        except Exception as e:
            logger.debug(f"Spectral feature extraction failed: {e}")
            for i in range(13):
                features[f"mfcc_{i+1}_mean"] = np.nan
                features[f"mfcc_{i+1}_std"] = np.nan
                features[f"delta_mfcc_{i+1}_mean"] = np.nan
            for key in [
                "spectral_centroid_mean", "spectral_centroid_std",
                "spectral_bandwidth_mean", "spectral_rolloff_mean",
                "zcr_mean", "zcr_std",
            ]:
                features[key] = np.nan

        return features


# =============================================================================
# 2. VOWEL SPACE ANALYSIS (VSA / VAI / FCR)
# =============================================================================

class VowelSpaceAnalyzer:
    """
    Computes Vowel Space Area (VSA), Vowel Articulation Index (VAI),
    and Formant Centralization Ratio (FCR) from formant frequencies.

    Reference:
        Sapir et al. (2010). "Formant Centralization Ratio: A Proposal for
        a New Acoustic Measure of Dysarthric Speech."
    """

    @staticmethod
    def compute_vsa(formant_coords: dict) -> float:
        """Compute Vowel Space Area using Convex Hull on the F1-F2 plane.

        Args:
            formant_coords: Dict with vowel keys, each containing 'f1' and 'f2'.
                Example: {'a': {'f1': 800, 'f2': 1200}, 'i': {'f1': 300, 'f2': 2300}, ...}

        Returns:
            Vowel Space Area in Hz^2.
        """
        points = np.array([[v["f1"], v["f2"]] for v in formant_coords.values()])
        if len(points) < 3:
            return np.nan
        try:
            hull = ConvexHull(points)
            return hull.volume  # In 2D, 'volume' gives area
        except Exception:
            return np.nan

    @staticmethod
    def compute_vai(formant_coords: dict) -> float:
        """Compute Vowel Articulation Index (VAI).

        VAI = (F2_i + F1_a) / (F1_i + F1_u + F2_u + F2_a)

        Args:
            formant_coords: Must contain keys 'a', 'i', 'u' with 'f1' and 'f2'.

        Returns:
            VAI value.
        """
        try:
            f1_a = formant_coords["a"]["f1"]
            f2_a = formant_coords["a"]["f2"]
            f1_i = formant_coords["i"]["f1"]
            f2_i = formant_coords["i"]["f2"]
            f1_u = formant_coords["u"]["f1"]
            f2_u = formant_coords["u"]["f2"]
            return (f2_i + f1_a) / (f1_i + f1_u + f2_u + f2_a)
        except (KeyError, ZeroDivisionError):
            return np.nan

    @staticmethod
    def compute_fcr(formant_coords: dict) -> float:
        """Compute Formant Centralization Ratio (FCR = 1 / VAI)."""
        vai = VowelSpaceAnalyzer.compute_vai(formant_coords)
        if vai and vai != 0 and not np.isnan(vai):
            return 1.0 / vai
        return np.nan


# =============================================================================
# 3. DATASET LOADERS
# =============================================================================

class VOICEDLoader:
    """
    Loads the VOICED dataset from PhysioNet.

    Dataset structure:
        voiced/
            RECORDS             # List of record names
            voiceNNN.dat        # Binary signal (WFDB format)
            voiceNNN.hea        # Header with metadata (age, sex, diagnosis)
            voiceNNN.txt        # Signal samples in plain text
            voiceNNN-info.txt   # Extended demographics and clinical metadata

    Reference:
        Cesari et al. (2018). "A new database of healthy and pathological voices."
        Computers & Electrical Engineering, 68, 310-321.

    Download:
        https://physionet.org/content/voiced/1.0.0/
    """

    DIAGNOSIS_MAP = {
        "healthy": 0,
        "hyperkinetic dysphonia": 1,
        "hypokinetic dysphonia": 1,
        "reflux laryngitis": 1,
    }
    LABEL_NAMES = {0: "Healthy", 1: "Pathological"}

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> pd.DataFrame:
        """Load all records and extract features.

        Returns:
            DataFrame with features and labels.
        """
        extractor = AcousticFeatureExtractor(target_sr=16000)
        records = self._get_record_list()
        all_features = []

        for idx, rec_name in enumerate(records):
            logger.info(f"Processing VOICED record {idx+1}/{len(records)}: {rec_name}")
            try:
                signal, sr, metadata = self._load_record(rec_name)
                feats = extractor.extract_all_features(signal, sr)
                feats["record_id"] = rec_name
                feats["age"] = metadata.get("age", np.nan)
                feats["sex"] = metadata.get("sex", "unknown")
                feats["diagnosis"] = metadata.get("diagnosis", "unknown")

                # Binary label: 0 = Healthy, 1 = Pathological
                diag = metadata.get("diagnosis", "").lower().strip()
                feats["label"] = self.DIAGNOSIS_MAP.get(diag, 1)

                all_features.append(feats)
            except Exception as e:
                logger.warning(f"Failed to process {rec_name}: {e}")

        return pd.DataFrame(all_features)

    def _get_record_list(self) -> list:
        """Read RECORDS file to get list of record names."""
        records_file = self.data_dir / "RECORDS"
        if records_file.exists():
            with open(records_file) as f:
                return [line.strip() for line in f if line.strip()]

        # Fallback: find .hea files
        hea_files = sorted(self.data_dir.glob("voice*.hea"))
        return [f.stem for f in hea_files]

    def _load_record(self, rec_name: str) -> tuple:
        """Load a single VOICED record.

        Returns:
            (signal, sample_rate, metadata_dict)
        """
        # Try WFDB first
        try:
            import wfdb
            record = wfdb.rdrecord(str(self.data_dir / rec_name))
            signal = record.p_signal.flatten()
            sr = record.fs
            metadata = self._parse_hea_comments(record.comments)
            return signal, sr, metadata
        except ImportError:
            pass

        # Fallback: load from .txt file
        txt_path = self.data_dir / f"{rec_name}.txt"
        signal = np.loadtxt(str(txt_path))
        sr = 8000  # VOICED default

        # Parse metadata from .hea file
        metadata = {}
        hea_path = self.data_dir / f"{rec_name}.hea"
        if hea_path.exists():
            with open(hea_path) as f:
                for line in f:
                    if line.startswith("#"):
                        metadata = self._parse_hea_line(line)
                        break
        return signal, sr, metadata

    @staticmethod
    def _parse_hea_comments(comments: list) -> dict:
        """Parse metadata from WFDB header comments."""
        metadata = {}
        for comment in comments:
            age_m = re.search(r"<age>:\s*(\d+)", comment)
            sex_m = re.search(r"<sex>:\s*(\w)", comment)
            diag_m = re.search(r"<diagnoses>:\s*(.+?)(?:\s*<medications>|$)", comment)
            if age_m:
                metadata["age"] = int(age_m.group(1))
            if sex_m:
                metadata["sex"] = sex_m.group(1)
            if diag_m:
                metadata["diagnosis"] = diag_m.group(1).strip()
        return metadata

    @staticmethod
    def _parse_hea_line(line: str) -> dict:
        """Parse a single .hea comment line."""
        metadata = {}
        age_m = re.search(r"<age>:\s*(\d+)", line)
        sex_m = re.search(r"<sex>:\s*(\w)", line)
        diag_m = re.search(r"<diagnoses>:\s*(.+?)(?:\s*<medications>|$)", line)
        if age_m:
            metadata["age"] = int(age_m.group(1))
        if sex_m:
            metadata["sex"] = sex_m.group(1)
        if diag_m:
            metadata["diagnosis"] = diag_m.group(1).strip()
        return metadata


class SVDLoader:
    """
    Loads the Saarbrucken Voice Database (SVD).

    Dataset structure (after download via svd-downloader):
        svd/
            data.json                       # Master metadata
            healthy/
                female/{speaker_id}/{session_id}/{file_id}.wav
                male/{speaker_id}/{session_id}/{file_id}.wav
            pathological/
                female/{speaker_id}/{session_id}/{file_id}.wav
                male/{speaker_id}/{session_id}/{file_id}.wav

    Specs: WAV 50 kHz, 16-bit. Vowels /a/, /i/, /u/ at normal/high/low pitch.

    Reference:
        Putzer, M. and Barry, W. "Saarbrucken Voice Database."
        Institute of Phonetics, Saarland University.

    Download:
        https://stimmdb.coli.uni-saarland.de/
        pip install svd-downloader && python -m svd-downloader /path/to/output
    """

    def __init__(self, data_dir: str, vowel: str = "a", pitch: str = "n"):
        """
        Args:
            data_dir: Root directory of downloaded SVD data.
            vowel: Which vowel to use ('a', 'i', or 'u').
            pitch: Which pitch variant ('n'=normal, 'h'=high, 'l'=low, 'lhl'=rising-falling).
        """
        self.data_dir = Path(data_dir)
        self.vowel = vowel
        self.pitch = pitch
        self.file_pattern = f"{vowel}_{pitch}"

    def load(self) -> pd.DataFrame:
        """Load SVD records and extract features.

        Returns:
            DataFrame with features and labels.
        """
        extractor = AcousticFeatureExtractor(target_sr=16000)
        all_features = []
        file_count = 0

        for label_name in ["healthy", "pathological"]:
            label = 0 if label_name == "healthy" else 1
            label_dir = self.data_dir / label_name

            if not label_dir.exists():
                logger.warning(f"Directory not found: {label_dir}")
                continue

            wav_files = sorted(label_dir.rglob("*.wav"))
            for wav_path in wav_files:
                # Filter by vowel and pitch if identifiable from filename
                fname = wav_path.stem.lower()
                if self.file_pattern and self.file_pattern not in fname:
                    continue

                file_count += 1
                logger.info(f"Processing SVD file {file_count}: {wav_path.name} ({label_name})")

                try:
                    import librosa
                    signal, sr = librosa.load(str(wav_path), sr=None)
                    feats = extractor.extract_all_features(signal, sr)
                    feats["file"] = wav_path.name
                    feats["label"] = label
                    feats["label_name"] = label_name

                    # Extract gender from directory structure
                    parts = wav_path.relative_to(label_dir).parts
                    if parts:
                        feats["gender"] = parts[0] if parts[0] in ("male", "female") else "unknown"

                    all_features.append(feats)
                except Exception as e:
                    logger.warning(f"Failed to process {wav_path}: {e}")

        logger.info(f"Loaded {len(all_features)} SVD samples")
        return pd.DataFrame(all_features)


class GenericWavLoader:
    """
    Generic loader for WAV-based datasets (FEMH, custom datasets).

    Expected structure:
        data_dir/
            healthy/     *.wav files
            pathological/ *.wav files
        OR
        data_dir/
            *.wav files + labels.csv (with columns: filename, label)
    """

    def __init__(self, data_dir: str, labels_csv: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.labels_csv = labels_csv

    def load(self) -> pd.DataFrame:
        """Load WAV files and extract features."""
        extractor = AcousticFeatureExtractor(target_sr=16000)
        all_features = []

        if self.labels_csv:
            return self._load_from_csv(extractor)

        # Try directory-based label structure
        for label_name, label_val in [("healthy", 0), ("pathological", 1),
                                       ("normal", 0), ("pathology", 1)]:
            label_dir = self.data_dir / label_name
            if not label_dir.exists():
                continue
            for wav_path in sorted(label_dir.glob("*.wav")):
                logger.info(f"Processing: {wav_path.name}")
                try:
                    import librosa
                    signal, sr = librosa.load(str(wav_path), sr=None)
                    feats = extractor.extract_all_features(signal, sr)
                    feats["file"] = wav_path.name
                    feats["label"] = label_val
                    all_features.append(feats)
                except Exception as e:
                    logger.warning(f"Failed: {wav_path}: {e}")

        return pd.DataFrame(all_features)

    def _load_from_csv(self, extractor: AcousticFeatureExtractor) -> pd.DataFrame:
        """Load using a CSV label file."""
        labels_df = pd.read_csv(self.labels_csv)
        all_features = []

        for _, row in labels_df.iterrows():
            wav_path = self.data_dir / row["filename"]
            if not wav_path.exists():
                continue
            try:
                import librosa
                signal, sr = librosa.load(str(wav_path), sr=None)
                feats = extractor.extract_all_features(signal, sr)
                feats["file"] = row["filename"]
                feats["label"] = row["label"]
                all_features.append(feats)
            except Exception as e:
                logger.warning(f"Failed: {wav_path}: {e}")

        return pd.DataFrame(all_features)


# =============================================================================
# 4. RANDOM FOREST CLASSIFIER PIPELINE
# =============================================================================

class VoicePathologyClassifier:
    """
    Random Forest classifier for voice pathology detection.

    Pipeline:
        1. Impute missing values (median)
        2. Standardize features
        3. Random Forest classification
    """

    FEATURE_COLUMNS = None  # Set during training

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.pipeline = None
        self.best_params = None
        self.feature_importances = None

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Separate features from labels and metadata.

        Returns:
            (X, y, feature_names)
        """
        # Identify numeric feature columns (exclude metadata and labels)
        exclude_cols = {"record_id", "file", "label", "label_name", "diagnosis",
                        "sex", "gender", "age", "vowel"}
        feature_cols = [c for c in df.columns
                        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

        self.FEATURE_COLUMNS = feature_cols
        X = df[feature_cols].values
        y = df["label"].values

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y, feature_cols

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        test_size: float = 0.15,
        cv_folds: int = 5,
        tune_hyperparams: bool = True,
    ) -> dict:
        """Train Random Forest with optional hyperparameter tuning and evaluate.

        Args:
            X: Feature matrix.
            y: Label vector.
            feature_names: Names of features.
            test_size: Fraction for test set.
            cv_folds: Number of cross-validation folds.
            tune_hyperparams: Whether to run grid search.

        Returns:
            Dictionary with evaluation results.
        """
        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        # Build pipeline
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ])

        # Hyperparameter tuning
        if tune_hyperparams:
            param_grid = {
                "clf__n_estimators": [100, 300, 500],
                "clf__max_depth": [None, 10, 20, 30],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", "log2"],
            }
            logger.info("Running hyperparameter search (this may take a while)...")
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f"Best params: {self.best_params}")
        else:
            self.pipeline.fit(X_train, y_train)

        # Cross-validation scores
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, scoring="accuracy")

        # Predictions
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)

        # Feature importance
        rf_model = self.pipeline.named_steps["clf"]
        self.feature_importances = pd.DataFrame({
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        # Compute metrics
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "feature_importances": self.feature_importances,
            "best_params": self.best_params,
            "X_test": X_test,
        }

        # AUC-ROC (binary classification)
        if len(np.unique(y)) == 2:
            results["auc_roc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            results["auc_roc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")

        return results


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

class ResultVisualizer:
    """Generates visualizations for classification results."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_all(self, results: dict, feature_names: list, label_names: dict = None):
        """Generate all plots and save to output directory."""
        if label_names is None:
            label_names = {0: "Healthy", 1: "Pathological"}

        self.plot_confusion_matrix(results, label_names)
        self.plot_feature_importance(results, top_n=20)
        self.plot_roc_curve(results, label_names)
        self.plot_cv_summary(results)
        logger.info(f"All plots saved to {self.output_dir}/")

    def plot_confusion_matrix(self, results: dict, label_names: dict):
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = results["confusion_matrix"]
        display_labels = [label_names.get(i, str(i)) for i in range(cm.shape[0])]
        disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Confusion Matrix - Voice Pathology Classification\n(Random Forest)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_feature_importance(self, results: dict, top_n: int = 20):
        """Plot top N most important features."""
        fi = results["feature_importances"].head(top_n)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=fi, x="importance", y="feature", ax=ax, palette="viridis")
        ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
        ax.set_xlabel("Importance (Mean Decrease in Impurity)")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_roc_curve(self, results: dict, label_names: dict):
        """Plot ROC curve (binary classification)."""
        y_test = results["y_test"]
        y_proba = results["y_proba"]

        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            auc = results["auc_roc"]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
            ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random baseline")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve - Voice Pathology Detection\n(Random Forest)")
            ax.legend(loc="lower right")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            plt.tight_layout()
            plt.savefig(self.output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
            plt.close()

    def plot_cv_summary(self, results: dict):
        """Plot cross-validation accuracy summary."""
        fig, ax = plt.subplots(figsize=(6, 4))
        metrics = {
            "Test Accuracy": results["accuracy"],
            "CV Mean Accuracy": results["cv_accuracy_mean"],
            "F1 (Weighted)": results["f1_weighted"],
            "AUC-ROC": results["auc_roc"],
        }
        bars = ax.bar(metrics.keys(), metrics.values(), color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"])
        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Summary")
        ax.set_ylim([0, 1.15])
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_summary.png", dpi=150, bbox_inches="tight")
        plt.close()


# =============================================================================
# 6. DEMO MODE (synthetic data for testing the pipeline)
# =============================================================================

def generate_demo_dataset(n_samples: int = 300, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic voice features for pipeline demonstration.

    Creates realistic-looking acoustic features based on published statistics
    for healthy and pathological voices.

    Reference ranges from:
        - Teixeira & Fernandes (2014), voice pathology feature analysis
        - Al-Nasheri et al. (2017), MDVP parameters on SVD
    """
    rng = np.random.RandomState(random_state)
    n_healthy = n_samples // 2
    n_pathological = n_samples - n_healthy

    records = []
    for i in range(n_samples):
        is_pathological = i >= n_healthy
        label = 1 if is_pathological else 0

        if is_pathological:
            # Pathological voice characteristics
            f0_mean = rng.normal(170, 50)       # More variable F0
            f0_std = rng.normal(25, 15)         # Higher F0 variability
            f1_mean = rng.normal(680, 100)
            f2_mean = rng.normal(1400, 200)
            f3_mean = rng.normal(2500, 300)
            jitter = rng.normal(0.025, 0.015)   # Higher jitter
            shimmer = rng.normal(0.08, 0.04)    # Higher shimmer
            hnr = rng.normal(12, 6)             # Lower HNR
            cpps = rng.normal(5.0, 2.5)         # Lower CPPS
        else:
            # Healthy voice characteristics
            f0_mean = rng.normal(150, 30)
            f0_std = rng.normal(8, 4)
            f1_mean = rng.normal(750, 80)
            f2_mean = rng.normal(1500, 150)
            f3_mean = rng.normal(2600, 250)
            jitter = rng.normal(0.008, 0.004)
            shimmer = rng.normal(0.03, 0.015)
            hnr = rng.normal(22, 4)
            cpps = rng.normal(10.0, 2.0)

        # Generate correlated MFCC features
        mfccs_mean = rng.normal(0, 5, 13) + (2 if is_pathological else -2) * rng.random(13)
        mfccs_std = np.abs(rng.normal(3, 1, 13))
        delta_mfccs_mean = rng.normal(0, 1, 13)

        record = {
            "f0_mean": max(f0_mean, 50),
            "f0_std": abs(f0_std),
            "f0_min": max(f0_mean - abs(rng.normal(30, 10)), 50),
            "f0_max": f0_mean + abs(rng.normal(30, 10)),
            "f0_range": abs(rng.normal(40, 20)),
            "f0_median": f0_mean + rng.normal(0, 5),
            "f1_mean": max(f1_mean, 200),
            "f1_std": abs(rng.normal(50, 20)),
            "f2_mean": max(f2_mean, 800),
            "f2_std": abs(rng.normal(80, 30)),
            "f3_mean": max(f3_mean, 1500),
            "f3_std": abs(rng.normal(100, 40)),
            "jitter_local": max(jitter, 0.001),
            "jitter_local_abs": max(jitter * 0.001, 0.000001),
            "jitter_rap": max(jitter * 0.5, 0.0005),
            "jitter_ppq5": max(jitter * 0.6, 0.0006),
            "shimmer_local": max(shimmer, 0.005),
            "shimmer_local_db": max(shimmer * 5, 0.02),
            "shimmer_apq3": max(shimmer * 0.4, 0.002),
            "shimmer_apq5": max(shimmer * 0.5, 0.003),
            "hnr_mean": hnr,
            "hnr_std": abs(rng.normal(3, 1)),
            "cpps": max(cpps, 0.5),
            "spectral_centroid_mean": rng.normal(1500, 400) + (200 if is_pathological else 0),
            "spectral_centroid_std": abs(rng.normal(300, 100)),
            "spectral_bandwidth_mean": rng.normal(2000, 500),
            "spectral_rolloff_mean": rng.normal(3000, 800),
            "zcr_mean": rng.normal(0.06, 0.02) + (0.02 if is_pathological else 0),
            "zcr_std": abs(rng.normal(0.01, 0.005)),
            "label": label,
        }

        # Add MFCCs
        for j in range(13):
            record[f"mfcc_{j+1}_mean"] = mfccs_mean[j]
            record[f"mfcc_{j+1}_std"] = mfccs_std[j]
            record[f"delta_mfcc_{j+1}_mean"] = delta_mfccs_mean[j]

        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# 7. MAIN ENTRY POINT
# =============================================================================

def print_results(results: dict):
    """Print classification results to console."""
    print("\n" + "=" * 70)
    print("  VOICE PATHOLOGY CLASSIFICATION RESULTS (Random Forest)")
    print("=" * 70)

    print(f"\n  Test Accuracy:       {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  F1-Score (weighted): {results['f1_weighted']:.4f}")
    print(f"  AUC-ROC:             {results['auc_roc']:.4f}")
    print(f"  CV Accuracy:         {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std']:.4f})")

    if results["best_params"]:
        print(f"\n  Best Hyperparameters:")
        for k, v in results["best_params"].items():
            print(f"    {k}: {v}")

    print(f"\n  Classification Report:")
    print(results["classification_report"])

    print(f"\n  Top 10 Most Important Features:")
    print(results["feature_importances"].head(10).to_string(index=False))
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Voice Pathology Detection using Random Forest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with synthetic data (no dataset required)
  python voice_pathology_random_forest.py --mode demo

  # Classify using VOICED dataset from PhysioNet
  python voice_pathology_random_forest.py --mode voiced --data-dir /path/to/voiced

  # Classify using SVD dataset
  python voice_pathology_random_forest.py --mode svd --data-dir /path/to/svd --vowel a

  # Classify using any WAV-based dataset
  python voice_pathology_random_forest.py --mode generic --data-dir /path/to/wavs

  # Use pre-extracted features from a CSV file
  python voice_pathology_random_forest.py --mode csv --features-csv features.csv
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="demo",
        choices=["demo", "voiced", "svd", "generic", "csv"],
        help="Data loading mode (default: demo)",
    )
    parser.add_argument("--data-dir", type=str, help="Path to dataset directory")
    parser.add_argument("--features-csv", type=str, help="Path to pre-extracted features CSV")
    parser.add_argument("--labels-csv", type=str, help="Path to labels CSV (for generic mode)")
    parser.add_argument("--vowel", type=str, default="a", help="Vowel for SVD (a, i, u)")
    parser.add_argument("--pitch", type=str, default="n", help="Pitch for SVD (n, h, l, lhl)")
    parser.add_argument("--n-samples", type=int, default=300, help="Samples for demo mode")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for plots")
    parser.add_argument("--save-features", type=str, help="Save extracted features to CSV")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Voice Pathology Detection - Random Forest Classifier")
    print("  Thesis: Automated Pre/Post-Surgical Voice Change Assessment")
    print("=" * 70)

    # ---- Load or generate data ----
    if args.mode == "demo":
        print(f"\n[INFO] Running in DEMO mode with {args.n_samples} synthetic samples.")
        print("[INFO] This demonstrates the full pipeline without requiring a dataset.")
        print("[INFO] For real data, use --mode voiced/svd/generic/csv\n")
        df = generate_demo_dataset(n_samples=args.n_samples, random_state=args.seed)

    elif args.mode == "voiced":
        if not args.data_dir:
            print("[ERROR] --data-dir required for VOICED mode")
            print("[INFO]  Download from: https://physionet.org/content/voiced/1.0.0/")
            return
        print(f"\n[INFO] Loading VOICED dataset from {args.data_dir}")
        loader = VOICEDLoader(args.data_dir)
        df = loader.load()

    elif args.mode == "svd":
        if not args.data_dir:
            print("[ERROR] --data-dir required for SVD mode")
            print("[INFO]  Download via: pip install svd-downloader")
            return
        print(f"\n[INFO] Loading SVD dataset from {args.data_dir} (vowel={args.vowel}, pitch={args.pitch})")
        loader = SVDLoader(args.data_dir, vowel=args.vowel, pitch=args.pitch)
        df = loader.load()

    elif args.mode == "generic":
        if not args.data_dir:
            print("[ERROR] --data-dir required for generic mode")
            return
        print(f"\n[INFO] Loading WAV files from {args.data_dir}")
        loader = GenericWavLoader(args.data_dir, labels_csv=args.labels_csv)
        df = loader.load()

    elif args.mode == "csv":
        if not args.features_csv:
            print("[ERROR] --features-csv required for CSV mode")
            return
        print(f"\n[INFO] Loading pre-extracted features from {args.features_csv}")
        df = pd.read_csv(args.features_csv)

    if df.empty:
        print("[ERROR] No data loaded. Check your data directory.")
        return

    print(f"[INFO] Dataset: {len(df)} samples loaded")
    print(f"[INFO] Label distribution: {df['label'].value_counts().to_dict()}")

    # Save extracted features if requested
    if args.save_features:
        df.to_csv(args.save_features, index=False)
        print(f"[INFO] Features saved to {args.save_features}")

    # ---- Train and evaluate ----
    classifier = VoicePathologyClassifier(n_estimators=300, random_state=args.seed)
    X, y, feature_names = classifier.prepare_data(df)

    results = classifier.train_and_evaluate(
        X, y, feature_names,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        tune_hyperparams=not args.no_tune,
    )

    # ---- Print results ----
    print_results(results)

    # ---- Generate plots ----
    visualizer = ResultVisualizer(output_dir=args.output_dir)
    visualizer.plot_all(results, feature_names)

    print(f"\n[INFO] Plots saved to {args.output_dir}/")
    print("[INFO] Files: confusion_matrix.png, feature_importance.png, roc_curve.png, performance_summary.png")
    print("\n[DONE] Pipeline completed successfully.\n")


if __name__ == "__main__":
    main()
