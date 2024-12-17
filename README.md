# Analyse Comparative de signaux audio avec Streamlit

Ce projet utilise **Streamlit** pour analyser et comparer les bruits émis par deux compresseurs en utilisant des fichiers audio `.wav` (fonctionne avec tout type de signaux echantilloné à 44.1 kHz). L'application affiche la densité spectrale de puissance (DSP), compare les spectres des deux signaux, et effectue des tests statistiques pour détecter les différences significatives. De plus, les fréquences dominantes des signaux sont extraites et affichées, ce qui permet de détecter d'éventuelles différences dans les caractéristiques du bruit des compresseurs.

## Fonctionnalités

- **Comparaison des signaux audio** : Affiche les densités spectrales de puissance (DSP) de deux fichiers `.wav` dans une plage de 0 à 10 kHz.
- **Tests statistiques** : Compare les spectres des deux fichiers audio à l'aide du test de Kolmogorov-Smirnov et du test t de Student.
- **Fréquences dominantes** : Identifie et affiche les fréquences dominantes de chaque signal, avec des informations visuelles mises en avant.
- **Signaux temporels** : Affiche les signaux audio sous forme de graphiques temporels pour une comparaison visuelle des signaux.

## Utilisation

- **Lien vers l'URL :** https://noise-comp-dom.streamlit.app/








### Installation avec `requirements.txt` si vous voulez l'utiliser en *local*


Clonez ce dépôt et installez les dépendances avec `pip` :

```bash
git clone https://github.com/your-username/compresser-noise-analysis.git
cd compresser-noise-analysis
pip install -r requirements.txt

  
### Prérequis

Assurez-vous que vous avez installé les dépendances suivantes :

- **Python 3.7 ou supérieur**
- **Streamlit** pour l'interface web interactive
- **Librosa** pour le traitement audio
- **Scipy**, **Matplotlib**, et **Numpy** pour l'analyse et la visualisation des données
