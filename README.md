# SDN Network DDoS Detection Using Machine Learning 🦊

Bienvenue dans notre projet SDN pour la détection des attaques DDoS avec Machine Learning ! Ce projet utilise **Ryu** comme contrôleur SDN et **Mininet** pour simuler un réseau, le tout en intégrant un modèle de Machine Learning pour détecter et analyser les attaques DDoS. C'est un peu comme un renard rusé qui détecte les prédateurs avant qu'ils n'attaquent (avec un peu de code et de data, bien sûr 😎).

## Pré-requis 🛠️
Avant de commencer à jouer avec le réseau, voici les outils dont tu auras besoin :

- **VirtualBox** (ou n'importe quelle autre plateforme de virtualisation)
- **Python 3.9** (Si tu n'as pas Python 3.9, l'install de Ryu va te faire grincer des dents 🦷)
- **Ryu Controller** (le cerveau de ton réseau SDN)
- **Mininet** (pour simuler un réseau et lancer les attaques)
- **Machine Learning** pour la détection des attaques DDoS

### Lien vers les ressources 🎁
- [**Dataset** pour les attaques DDoS](https://drive.google.com/file/d/1N2QLDPb90XOdxcuQ_Fb7ZSVOG4J3w_zY/view?usp=sharing)
- [**Ryu Controller VM**](https://drive.google.com/file/d/1_5PQWBsQcVnxtzwhUMzP-w2mR9MZrG6S/view?usp=sharing)
- [**Mininet VM**](https://drive.google.com/file/d/1H7Hs-yruNQKMDmcdgHJGHIDtopPNFAvH/view?usp=sharing)

## Installation 🚀

### 1. Importer les machines virtuelles

Avant toute chose, importe les VM dans **VirtualBox** (ou un autre logiciel de virtualisation de ton choix).

### 2. Modifier l'adresse IP du contrôleur Ryu

Le contrôleur Ryu a besoin d'une configuration spécifique, notamment l'adresse IP du contrôleur dans le code source. Modifie l'adresse dans le script pour qu'elle corresponde à celle de ta machine virtuelle Ryu.

### 3. Lancer le contrôleur Ryu

Ouvre un terminal dans ta VM Ryu et exécute la commande suivante pour démarrer le contrôleur :

```bash
ryu-manager DT_controller.py
```

### 4. Lancer Mininet

Dans ta VM Mininet, démarre le réseau avec le script `topology.py` :

```bash
sudo python topology.py
```

### 5. Lancer les attaques DDoS

Les attaques DDoS sont simulées dans cet environnement. Pour voir les résultats, suis les instructions des vidéos. Simule les attaques DDoS et observe les résultats du modèle de Machine Learning.

## Comment ça marche ? 🤔

- **Ryu** : Le contrôleur SDN gère le réseau et configure les commutateurs à la volée. C'est le cerveau de tout le réseau.
- **Mininet** : Utilisé pour créer une topologie réseau virtuelle. Nous y lançons les attaques DDoS.
- **Machine Learning** : On utilise un modèle pour analyser le trafic réseau et détecter les attaques DDoS en temps réel.

Le tout fonctionne dans une architecture SDN, ce qui permet une gestion centralisée et une détection des attaques en temps réel. C'est un peu comme avoir un renard en alerte 24/7 pour éviter les attaques de loup 😜.

## Dépannage 🦊

### Erreur commune #1 : Version de Python

Si tu rencontres une erreur liée à la version de Python, assure-toi que tu utilises **Python 3.9**. Si tu n'as pas cette version, voici comment la compiler :

1. Télécharge Python 3.9 via `wget` :

   ```bash
   wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz
   tar -xvf Python-3.9.0.tgz
   cd Python-3.9.0
   ./configure
   make altinstall
   ```

2. Ensuite, définis Python 3 comme l'alias par défaut :

   ```bash
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1
   ```

### Erreur commune #2 : Problème d'importation de librairies

Si le système te dit que certaines librairies ne sont pas installées, exécute :

```bash
pip install -r requirements.txt
```

Si tu as des problèmes spécifiques avec les versions de `evenlet`, essaie de désinstaller la version actuelle et d'installer la version **0.30.2** :

```bash
pip uninstall eventlet
pip install eventlet==0.30.2
```

### Erreur commune #3 : Ryu ne démarre pas

Assure-toi que tu as bien configuré l'adresse IP du contrôleur et que tous les ports nécessaires sont ouverts dans ton réseau virtuel.

## À propos 🦊

Ce projet a été développé par **Fox** 🦊 (oui, c’est moi). J'ai décidé de combiner mon amour pour les réseaux, la cybersécurité et le Machine Learning pour créer quelque chose d’unique. Si tu as aimé ce projet, n’hésite pas à me le dire (ou même à m’envoyer des memes de renard). Et si tu veux contribuer, tu sais où me trouver.

