import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal as signal
from scipy.stats import ks_2samp, ttest_ind
from io import BytesIO

# Titre de l'application
st.title("Analyse Comparative du bruit de 2 compresseurs 🔊")

# Téléchargement des deux fichiers WAV
uploaded_file_1 = st.file_uploader("Télécharger le bruit du 1er compresseur", type=["wav"])
uploaded_file_2 = st.file_uploader("Télécharger le bruit du 2e compresseur", type=["wav"])

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    try:
        # Chargement des fichiers audio avec la même fréquence d'échantillonnage (44.1kHz)
        y1, sr1 = librosa.load(BytesIO(uploaded_file_1.getvalue()), sr=44100)
        y2, sr2 = librosa.load(BytesIO(uploaded_file_2.getvalue()), sr=44100)

        # Vérification de la durée des fichiers (les rendre égaux si nécessaire)
        min_len = min(len(y1), len(y2))
        y1, y2 = y1[:min_len], y2[:min_len]

        # Affichage des informations sur les fichiers audio
        filename_1 = uploaded_file_1.name.replace(".wav", "")
        filename_2 = uploaded_file_2.name.replace(".wav", "")
        duration_1 = librosa.get_duration(y=y1, sr=sr1)
        duration_2 = librosa.get_duration(y=y2, sr=sr2)
        st.write(f"{filename_1} - Durée : {duration_1:.2f} secondes, Fréquence d'échantillonnage : {sr1} Hz")
        st.write(f"{filename_2} - Durée : {duration_2:.2f} secondes, Fréquence d'échantillonnage : {sr2} Hz")

        # --- 1. Densité spectrale de puissance (PSD) entre 0 et 10 kHz ---
        f1, Pxx1 = signal.welch(y1, fs=sr1, nperseg=1024)
        f2, Pxx2 = signal.welch(y2, fs=sr2, nperseg=1024)

        # Limiter la plage à 0-10 kHz
        f1, Pxx1 = f1[f1 <= 10000], Pxx1[:len(f1[f1 <= 10000])]
        f2, Pxx2 = f2[f2 <= 10000], Pxx2[:len(f2[f2 <= 10000])]

        # Tracer les PSD côte à côte avec la même échelle
        max_y = max(np.max(Pxx1), np.max(Pxx2))
        st.subheader(f"🎧 Graphiques de Densité Spectrale de Puissance - DSP de {filename_1} et {filename_2}")
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].semilogx(f1, Pxx1)
        axs[0].set_ylim(0, max_y)
        axs[0].set_title(f"PSD de {filename_1}")
        axs[0].set_xlabel("Fréquence (Hz)")
        axs[0].set_ylabel("DSP (dB/Hz)")

        axs[1].semilogx(f2, Pxx2)
        axs[1].set_ylim(0, max_y)
        axs[1].set_title(f"PSD de {filename_2}")
        axs[1].set_xlabel("Fréquence (Hz)")
        axs[1].set_ylabel("DSP (dB/Hz)")

        st.pyplot(fig)

        # --- 2. Tests statistiques ---
        ks_stat, ks_p_value = ks_2samp(Pxx1, Pxx2)
        t_stat, t_p_value = ttest_ind(Pxx1, Pxx2)
        st.write(f"Test de Kolmogorov-Smirnov : statistique = {ks_stat:.4f}, p-value = {ks_p_value:.4f}")
        st.write(f"Test t de Student : statistique = {t_stat:.4f}, p-value = {t_p_value:.4f}")

        # --- 3. Valeur RMS ---
        rms1 = np.sqrt(np.mean(y1**2))
        rms2 = np.sqrt(np.mean(y2**2))
        st.write(f"Valeur RMS du signal de {filename_1} : {rms1:.4f}")
        st.write(f"Valeur RMS du signal de {filename_2} : {rms2:.4f}")

        # --- 4. Signal temporel ---
        st.subheader(f"📈 Signal temporel de {filename_1}")
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y1, sr=sr1)
        plt.title(f"Signal temporel de {filename_1}")
        st.pyplot(plt)

        st.subheader(f"📈 Signal temporel de {filename_2}")
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y2, sr=sr2)
        plt.title(f"Signal temporel de {filename_2}")
        st.pyplot(plt)

        # --- 5. Fréquences dominantes ---
        def extract_fundamental_frequency(y, sr):
            fft_result = np.fft.fft(y)
            fft_freq = np.fft.fftfreq(len(y), 1/sr)
            magnitude = np.abs(fft_result)
            dominant_freq = np.abs(fft_freq[np.argmax(magnitude[1:])])
            return dominant_freq

        freq1 = extract_fundamental_frequency(y1, sr1)
        freq2 = extract_fundamental_frequency(y2, sr2)
        st.subheader("🔊 Fréquences Dominantes (Résonance)")
        st.write(f"{filename_1} : {freq1:.2f} Hz")
        st.write(f"{filename_2} : {freq2:.2f} Hz")

    except Exception as e:
        st.error("⚠️ Une erreur est survenue lors de l'analyse des fichiers.")
        st.error(e)
