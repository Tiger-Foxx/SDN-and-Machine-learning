#!/bin/bash

# Script d'installation de Ryu et Mininet avec Python 3.9 dans un environnement virtuel.
# Auteur : 🦊 Fox-style Installation Script

# --- Configuration de base ---
PYTHON_VERSION="3.9.0"
VIRTUALENV_NAME="ryuenv"
RYU_PACKAGE="ryu"
EVENTLET_VERSION="0.30.2"

# Fonction pour afficher des messages d'erreur et arrêter le script
error_exit() {
    echo "❌ [Erreur] $1"
    exit 1
}

# Demander les droits d'administrateur
if [ "$EUID" -ne 0 ]; then
    echo "⚠️ Ce script nécessite des privilèges root. Relancez avec 'sudo'."
    exit 1
fi

echo "✅ Droits administrateur vérifiés."

# --- Étape 1 : Mise à jour du système ---
echo "🔄 Mise à jour des paquets système..."
apt-get update || error_exit "Impossible de mettre à jour les paquets."
apt-get upgrade -y || error_exit "Impossible de mettre à jour les paquets système."

# --- Étape 2 : Installation des dépendances pour Python et Mininet ---
echo "🔧 Installation des dépendances système..."
apt-get install -y build-essential wget libffi-dev libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev python3-pip python3-venv || error_exit "Échec de l'installation des dépendances système."

# --- Étape 3 : Installation de Python 3.9 ---
echo "🐍 Téléchargement et installation de Python $PYTHON_VERSION..."
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz || error_exit "Échec du téléchargement de Python $PYTHON_VERSION."
tar -xvf Python-$PYTHON_VERSION.tgz || error_exit "Échec de l'extraction de Python."
cd Python-$PYTHON_VERSION || error_exit "Impossible de naviguer dans le répertoire de Python."

echo "🔨 Compilation de Python $PYTHON_VERSION..."
./configure --enable-optimizations || error_exit "Échec de la configuration de Python."
make -j$(nproc) || error_exit "Échec de la compilation de Python."
make altinstall || error_exit "Échec de l'installation de Python $PYTHON_VERSION."
cd .. && rm -rf Python-$PYTHON_VERSION*  # Nettoyer les fichiers temporaires

# Vérification de la version installée
python3.9 --version || error_exit "Python 3.9 n'a pas été installé correctement."

# --- Étape 4 : Configuration d'alias pour Python 3.9 ---
echo "🔗 Configuration de Python 3.9 comme alias par défaut pour python3..."
update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1 || error_exit "Échec de la configuration de l'alias Python."
python3 --version

# --- Étape 5 : Création d'un environnement virtuel ---
echo "🌱 Création de l'environnement virtuel '$VIRTUALENV_NAME'..."
python3 -m venv $VIRTUALENV_NAME || error_exit "Impossible de créer l'environnement virtuel."
source $VIRTUALENV_NAME/bin/activate || error_exit "Impossible d'activer l'environnement virtuel."

# --- Étape 6 : Installation de Mininet (application et bibliothèque) ---
echo "📦 Installation de Mininet (application et bibliothèque Python)..."
apt-get install -y mininet || error_exit "Impossible d'installer l'application Mininet."
pip install mininet || error_exit "Impossible d'installer la bibliothèque Mininet."

# --- Étape 7 : Installation de Ryu ---
echo "📦 Installation de Ryu..."
pip install $RYU_PACKAGE || error_exit "Impossible d'installer Ryu."
pip install eventlet==$EVENTLET_VERSION || error_exit "Impossible d'installer la version compatible d'eventlet."

# Vérification de l'installation de Ryu
ryu-manager --version || error_exit "Ryu n'a pas été installé correctement."

# --- Nettoyage final ---
echo "🧹 Nettoyage des paquets inutiles..."
apt-get autoremove -y
apt-get autoclean -y

# --- Étape finale ---
echo "🎉 Installation complète ! Voici les étapes importantes pour commencer :"
echo "1️⃣ Activez votre environnement virtuel avec : 'source $VIRTUALENV_NAME/bin/activate'"
echo "2️⃣ Lancez Ryu Manager avec : 'ryu-manager'"
echo "3️⃣ Utilisez Mininet en ligne de commande ou programmez vos topologies !"
echo "4️⃣ Rappelez-vous que 'eventlet==$EVENTLET_VERSION' est essentiel pour la compatibilité."
