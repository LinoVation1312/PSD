import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal as signal
from io import BytesIO

# Titre de l'application
st.title("Analyse Comparative du bruit de 2 compresseurs 🔊")

# Téléchargement des deux fichiers WAV
uploaded_file_1 = st.file_uploader("Télécharger le bruit du 1er compresseur", type=["wav"])
uploaded_file_2 = st.file_uploader("Télécharger le bruit du 2e compresseur", type=["wav"])

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    try:
        # Indicateur de progression
        with st.spinner('Chargement et traitement des fichiers audio...'):
            # Chargement des fichiers avec une taille d'échantillon optimisée
            y1, sr1 = librosa.load(BytesIO(uploaded_file_1.getvalue()), sr=44100, duration=20)
            y2, sr2 = librosa.load(BytesIO(uploaded_file_2.getvalue()), sr=44100, duration=20)

            # Ajustement à la même durée
            min_len = min(len(y1), len(y2))
            y1, y2 = y1[:min_len], y2[:min_len]

            # --- 1. Densité spectrale de puissance optimisée ---
            st.subheader("🎧 Densité Spectrale de Puissance (0-10 kHz)")

            # Calcul rapide de la PSD avec une fenêtre plus petite
            f1, Pxx1 = signal.welch(y1, fs=sr1, nperseg=4096)
            f2, Pxx2 = signal.welch(y2, fs=sr2, nperseg=4096)

            # Limiter à 0-10 kHz
            mask1 = f1 <= 10000
            mask2 = f2 <= 10000
            f1, Pxx1 = f1[mask1], Pxx1[mask1]
            f2, Pxx2 = f2[mask2], Pxx2[mask2]

            # Tracer les graphes
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            max_y = max(np.max(Pxx1), np.max(Pxx2))

            axs[0].semilogx(f1, Pxx1, color='tab:blue')
            axs[0].set_ylim(0, max_y)
            axs[0].set_title("PSD du 1er compresseur")
            axs[0].set_xlabel("Fréquence (Hz)")
            axs[0].set_ylabel("DSP (dB/Hz)")

            axs[1].semilogx(f2, Pxx2, color='tab:orange')
            axs[1].set_ylim(0, max_y)
            axs[1].set_title("PSD du 2e compresseur")
            axs[1].set_xlabel("Fréquence (Hz)")
            st.pyplot(fig)

            # --- 2. Extraction de la fréquence dominante optimisée ---
            def extract_fundamental_frequency(y, sr):
                # Calcul de la FFT
                fft_result = np.fft.rfft(y)
                fft_freq = np.fft.rfftfreq(len(y), 1/sr)
                magnitude = np.abs(fft_result)

                # Limiter la recherche de pic entre 20 Hz et 2 kHz
                mask = (fft_freq >= 20) & (fft_freq <= 2000)
                fft_freq_filtered = fft_freq[mask]
                magnitude_filtered = magnitude[mask]

                # Trouver la fréquence correspondant au pic le plus élevé
                dominant_freq = fft_freq_filtered[np.argmax(magnitude_filtered)]
                return dominant_freq

            freq1 = extract_fundamental_frequency(y1, sr1)
            freq2 = extract_fundamental_frequency(y2, sr2)

            # Affichage des fréquences dominantes
            st.subheader("🔊 Fréquences Dominantes (Résonance)")
            st.markdown(
                f"<h2 style='color:cyan; text-align:center;'>{uploaded_file_1.name} : {freq1:.2f} Hz</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h2 style='color:deepskyblue; text-align:center;'>{uploaded_file_2.name} : {freq2:.2f} Hz</h2>",
                unsafe_allow_html=True
            )

            # --- 3. Valeur RMS ---
            rms1 = np.sqrt(np.mean(y1**2))
            rms2 = np.sqrt(np.mean(y2**2))
            st.subheader("📊 Valeurs RMS des signaux")
            st.write(f"**{uploaded_file_1.name}** : {rms1:.4f}")
            st.write(f"**{uploaded_file_2.name}** : {rms2:.4f}")

            # --- 4. Affichage des signaux temporels ---
            st.subheader(f"📈 Signal temporel de {uploaded_file_1.name}")
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(y1, sr=sr1)
            plt.title(f"Signal temporel de {uploaded_file_1.name}")
            plt.xlabel("Temps (s)")
            plt.ylabel("Amplitude")
            st.pyplot(plt)

            st.subheader(f"📈 Signal temporel de {uploaded_file_2.name}")
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(y2, sr=sr2)
            plt.title(f"Signal temporel de {uploaded_file_2.name}")
            plt.xlabel("Temps (s)")
            plt.ylabel("Amplitude")
            st.pyplot(plt)

    except Exception as e:
        st.error("⚠️ Une erreur est survenue lors de l'analyse des fichiers.")
        st.error(e)
