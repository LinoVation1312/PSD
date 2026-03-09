import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal as signal
from io import BytesIO
import os

st.set_page_config(
    page_title="Analyse Comparative du bruit",
    page_icon="https://icon-library.com/images/air-compressor-icon/air-compressor-icon-5.jpg",
    layout="centered"
)

st.title("Analyse Comparative du bruit de 2 compresseurs 🔊")

st.markdown("""
[Convertir les fichiers dictaphone iPhone en .wav ici](https://convertio.co/fr/)  
Cliquez sur le lien pour convertir vos fichiers audio en format .wav avant de les télécharger dans l'application.
""")

uploaded_file_1 = st.file_uploader("Télécharger le bruit du 1er compresseur", type=["wav"])
uploaded_file_2 = st.file_uploader("Télécharger le bruit du 2e compresseur", type=["wav"])

# --- Options d'analyse ---
with st.expander("⚙️ Options d'analyse", expanded=False):
    normalize_rms = st.checkbox(
        "Normaliser par le niveau RMS",
        value=False,
        help="Compense les différences de distance au micro. Compare les formes spectrales indépendamment du niveau global."
    )
    freq_max = st.slider("Fréquence maximale affichée (Hz)", 1000, 20000, 10000, step=500)
    freq_ref = st.number_input(
        "Fréquence de référence pour Lp (Pa²/Hz)",
        value=4e-10,
        format="%.2e",
        help="Valeur de référence pour le calcul en dB. Par défaut : 4e-10 Pa²/Hz (référence acoustique standard)"
    )

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    try:
        with st.spinner('Chargement et traitement des fichiers audio...'):
            y1, sr1 = librosa.load(BytesIO(uploaded_file_1.getvalue()), sr=44100, duration=60)
            y2, sr2 = librosa.load(BytesIO(uploaded_file_2.getvalue()), sr=44100, duration=60)

            min_len = min(len(y1), len(y2))
            y1, y2 = y1[:min_len], y2[:min_len]

            file_name_1 = os.path.splitext(uploaded_file_1.name)[0]
            file_name_2 = os.path.splitext(uploaded_file_2.name)[0]

            # --- RMS ---
            rms1 = np.sqrt(np.mean(y1**2))
            rms2 = np.sqrt(np.mean(y2**2))

            # --- Calcul PSD ---
            f1, Pxx1 = signal.welch(y1, fs=sr1, nperseg=2**14)
            f2, Pxx2 = signal.welch(y2, fs=sr2, nperseg=2**14)

            # Normalisation RMS optionnelle
            if normalize_rms:
                Pxx1 = Pxx1 / (rms1**2)
                Pxx2 = Pxx2 / (rms2**2)

            # Passage en dB
            Pxx1_dB = 10 * np.log10(Pxx1 / freq_ref + 1e-12)
            Pxx2_dB = 10 * np.log10(Pxx2 / freq_ref + 1e-12)

            # Masque fréquentiel
            mask = f1 <= freq_max
            f_plot = f1[mask]
            P1_plot = Pxx1_dB[mask]
            P2_plot = Pxx2_dB[mask]
            diff_plot = P1_plot - P2_plot

        # ================================================================
        # GRAPHE 1 — Superposition des deux PSD
        # ================================================================
        st.subheader("🎧 Densité Spectrale de Puissance — Comparaison")

        norm_label = " (normalisé RMS)" if normalize_rms else ""

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.semilogx(f_plot, P1_plot, color='tab:blue', linewidth=1.2, label=file_name_1)
        ax1.semilogx(f_plot, P2_plot, color='tab:orange', linewidth=1.2, label=file_name_2)
        ax1.fill_between(f_plot, P1_plot, P2_plot, alpha=0.08, color='gray')
        ax1.set_xlabel("Fréquence (Hz)")
        ax1.set_ylabel(f"Lp (dB ref {freq_ref:.0e} Pa²/Hz){norm_label}")
        ax1.set_title("PSD superposées")
        ax1.legend()
        ax1.grid(True, which='both', color='gray', linestyle='--', linewidth=0.4)
        fig1.tight_layout()
        st.pyplot(fig1)

        st.markdown("""
        \n \n \n
        <p style="text-align: justify;">
            <strong>Défintion : La PSD (Densité Spectrale de Puissance)</strong> est une représentation du signal dans le 
            <strong>domaine fréquentiel</strong>, contrairement à l'analyse temporelle qui mesure l'amplitude du signal 
            à un instant donné. Elle montre <strong>comment la puissance du signal est répartie sur différentes fréquences</strong>. 
            En d'autres termes, la PSD permet de savoir quelle quantité de puissance est concentrée dans chaque plage de 
            fréquences, donnant ainsi une vue détaillée de l'énergie présente dans le signal à travers le temps et les fréquences.
        </p>
        """, unsafe_allow_html=True)

        # ================================================================
        # GRAPHE 2 — Différence spectrale
        # ================================================================
        st.subheader(f"📉 Différence spectrale : {file_name_1} − {file_name_2}")

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.semilogx(f_plot, diff_plot, color='tab:red', linewidth=1.0, label="Différence (dB)")
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.fill_between(f_plot, diff_plot, 0,
                         where=(diff_plot > 0), alpha=0.15, color='tab:blue',
                         label=f"{file_name_1} plus fort")
        ax2.fill_between(f_plot, diff_plot, 0,
                         where=(diff_plot < 0), alpha=0.15, color='tab:orange',
                         label=f"{file_name_2} plus fort")
        ax2.set_xlabel("Fréquence (Hz)")
        ax2.set_ylabel("ΔLp (dB)")
        ax2.set_title("Différence spectrale (positif = compresseur 1 dominant)")
        ax2.legend()
        ax2.grid(True, which='both', color='gray', linestyle='--', linewidth=0.4)
        fig2.tight_layout()
        st.pyplot(fig2)

        # ================================================================
        # MÉTRIQUES
        # ================================================================
        st.subheader("📊 Métriques globales")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMS — " + file_name_1, f"{rms1:.4f}")
            st.metric("RMS — " + file_name_2, f"{rms2:.4f}")
        with col2:
            lp1 = 10 * np.log10(rms1**2 / freq_ref + 1e-12)
            lp2 = 10 * np.log10(rms2**2 / freq_ref + 1e-12)
            st.metric("Lp global — " + file_name_1, f"{lp1:.1f} dB")
            st.metric("Lp global — " + file_name_2, f"{lp2:.1f} dB")
        with col3:
            delta_lp = lp1 - lp2
            freq_max_diff = f_plot[np.argmax(np.abs(diff_plot))]
            st.metric("ΔLp global", f"{delta_lp:+.1f} dB")
            st.metric("Fréq. d'écart max", f"{freq_max_diff:.0f} Hz")

        # ================================================================
        # FRÉQUENCES DOMINANTES
        # ================================================================
        st.subheader("🔊 Fréquences dominantes")

        def extract_dominant_frequency(y, sr):
            fft_result = np.fft.rfft(y)
            fft_freq = np.fft.rfftfreq(len(y), 1/sr)
            magnitude = np.abs(fft_result)
            mask = (fft_freq >= 20) & (fft_freq <= 2000)
            dominant_freq = fft_freq[mask][np.argmax(magnitude[mask])]
            return dominant_freq

        freq1 = extract_dominant_frequency(y1, sr1)
        freq2 = extract_dominant_frequency(y2, sr2)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(f"<h3 style='color:tab; text-align:center;'>{file_name_1}<br>{freq1:.1f} Hz</h3>",
                        unsafe_allow_html=True)
        with col_f2:
            st.markdown(f"<h3 style='color:orange; text-align:center;'>{file_name_2}<br>{freq2:.1f} Hz</h3>",
                        unsafe_allow_html=True)

        # ================================================================
        # SIGNAUX TEMPORELS
        # ================================================================
        with st.expander("📈 Afficher les signaux temporels"):
            fig3, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            librosa.display.waveshow(y1, sr=sr1, ax=axes[0], color='tab:blue')
            axes[0].set_title(f"Signal temporel — {file_name_1}")
            axes[0].set_ylabel("Amplitude")
            librosa.display.waveshow(y2, sr=sr2, ax=axes[1], color='tab:orange')
            axes[1].set_title(f"Signal temporel — {file_name_2}")
            axes[1].set_ylabel("Amplitude")
            axes[1].set_xlabel("Temps (s)")
            fig3.tight_layout()
            st.pyplot(fig3)

        # ================================================================
        # EXPORT PDF
        # ================================================================
        pdf_buffer = BytesIO()
        fig_export, axs_exp = plt.subplots(2, 1, figsize=(12, 9))
        axs_exp[0].semilogx(f_plot, P1_plot, color='tab:blue', linewidth=1.2, label=file_name_1)
        axs_exp[0].semilogx(f_plot, P2_plot, color='tab:orange', linewidth=1.2, label=file_name_2)
        axs_exp[0].fill_between(f_plot, P1_plot, P2_plot, alpha=0.08, color='gray')
        axs_exp[0].set_ylabel(f"Lp (dB)")
        axs_exp[0].set_title("PSD superposées")
        axs_exp[0].legend()
        axs_exp[0].grid(True, which='both', linestyle='--', linewidth=0.4)
        axs_exp[1].semilogx(f_plot, diff_plot, color='tab:red', linewidth=1.0)
        axs_exp[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axs_exp[1].fill_between(f_plot, diff_plot, 0, where=(diff_plot > 0), alpha=0.15, color='tab:blue')
        axs_exp[1].fill_between(f_plot, diff_plot, 0, where=(diff_plot < 0), alpha=0.15, color='tab:orange')
        axs_exp[1].set_xlabel("Fréquence (Hz)")
        axs_exp[1].set_ylabel("ΔLp (dB)")
        axs_exp[1].set_title("Différence spectrale")
        axs_exp[1].grid(True, which='both', linestyle='--', linewidth=0.4)
        fig_export.tight_layout()
        fig_export.savefig(pdf_buffer, format='pdf')
        pdf_buffer.seek(0)

        st.download_button(
            label="📥 Télécharger le rapport PSD en PDF",
            data=pdf_buffer,
            file_name="psd_comparaison.pdf",
            mime="application/pdf"
        )

        st.markdown("""
        ---
        Le code source (codé en Python) est disponible là :  
        [Code source sur GitHub](https://github.com/LinoVation1312/PSD)



        Gros Bisous



        Written by Lino Conord - Déc. 2024
        """)

    except Exception as e:
        st.error("⚠️ Une erreur est survenue lors de l'analyse des fichiers.")
        st.error(e)
