<!-- ================================================================== -->
<!--                       Challenge Collombole              -->
<!-- ================================================================== -->

# Collombole - Cahier des Charges

![Logo du Projet](https://www.encyclopedie-environnement.org/app/uploads/2017/10/collemboles_fig3_collembole-furca-tube-ventral.jpg "Logo du Projet")


Bienvenue sur le dépôt GitHub du projet **Collombole**. Ce document décrit l’ensemble des objectifs, des données, de la méthodologie et de la planification pour l’identification des collemboles via des techniques de deep learning.

---

## Table des Matières

- [Objectif](#objectif)
- [Données](#données)
  - [Données d'Entraînement](#données-dentrainement)
  - [Données de Test](#données-de-test)
- [Objectifs du Projet](#objectifs-du-projet)
  - [Labellisation des Imagettes de Test](#labellisation-des-imagettes-de-test)
  - [Correction des Labels](#correction-des-labels)
  - [Explicabilité du Modèle](#explicabilité-du-modèle)
- [Méthodologie Proposée](#méthodologie-proposée)
  - [Prétraitement des Données](#prétraitement-des-données)
  - [Sélection du Modèle](#sélection-du-modèle)
  - [Entraînement du Modèle](#entraînement-du-modèle)
  - [Correction des Labels](#correction-des-labels-1)
  - [Explicabilité du Modèle](#explicabilité-du-modèle-1)
- [Ressources et Outils](#ressources-et-outils)
- [Livrables](#livrables)
- [Planification](#planification)
- [Conclusion](#conclusion)

---

## Objectif

Le projet **Collombole** a pour but d’identifier des collemboles à partir d’images en utilisant des techniques de deep learning. Les objectifs principaux sont :
- **Labellisation des imagettes de test**
- **Correction des labels** dans la base d’entraînement en cas de désaccord entre experts
- **Explicabilité** des décisions du modèle à travers des statistiques et des visualisations

---

## Données

### Données d'Entraînement

- **Nombre d'images** : 1117 images de taille `3072 x 2048`
- **Format des labels** : `yolo+`, avec des annotations fournies par 4 experts  
- **Exemple de format** :

4_4_4_2 Ecopic 20180.58089192708333340.162597656250.35970052083333330.2587890625


### Données de Test

- **Nombre d'imagettes** : 1344 imagettes de taille variable
- **Base Kaggle** : 50 % public / 50 % privé

---

## Objectifs du Projet

### Labellisation des Imagettes de Test

- Fournir la labellisation pour chaque imagette de la base de test.
- **Catégories** : `AUTRE`, `Cer`, `CRY_THE`, `HYP_MAN`, `ISO_MIN`, `LEP`, `MET_AFF`, `PAR_NOT`, `FOND`.
- **Évaluation** : Utilisation de la métrique **F1-macro** sur Kaggle.

### Correction des Labels

- Proposer une correction des labels dans la base d'entraînement lors d’un désaccord entre les votants.

### Explicabilité du Modèle

- Fournir des statistiques et des visualisations pour expliquer les décisions du modèle.

---

## Méthodologie Proposée

### Prétraitement des Données

- **Redimensionnement des Images** : Adapter la taille des images aux modèles de deep learning.
- **Augmentation des Données** : Utiliser des techniques d’augmentation pour enrichir la diversité des données.
- **Normalisation** : Normaliser les valeurs des pixels pour une meilleure convergence du modèle.

### Sélection du Modèle

- **Architecture de Réseau** : Utilisation de CNN pré-entraînés (ex : ResNet, EfficientNet) ou de modèles spécifiques à la détection d’objets comme YOLO.
- **Transfer Learning** : Exploiter le transfert d'apprentissage sur des ensembles de données similaires.

### Entraînement du Modèle

- **Division des Données** : Séparer les données en ensembles d’entraînement, de validation et de test.
- **Optimisation** : Utilisation d’algorithmes comme Adam ou SGD avec des taux d’apprentissage adaptatifs.
- **Validation Croisée** : Mettre en place une validation croisée pour éviter le surapprentissage et évaluer la performance.

### Correction des Labels

- **Détection des Désaccords** : Identifier les points de désaccord entre les votants.
- **Correction Automatique** : Appliquer des algorithmes de consensus ou des modèles pour ajuster automatiquement les labels.

### Explicabilité du Modèle

- **Visualisation des Caractéristiques** : Utiliser des techniques telles que les cartes de chaleur pour mettre en évidence les zones d’intérêt du modèle.
- **Statistiques** : Fournir des indicateurs de performance (précision, rappel, F1-score) pour interpréter les décisions du modèle.

---

## Ressources et Outils

- **Matériel** : 
- GPU RTX 4 (1 GPU par groupe, 8 cœurs GPU par groupe, 545 RAM par groupe)
- **Logiciels** : 
- Python, TensorFlow/Keras, PyTorch, scikit-learn, OpenCV
- **Plateforme** : 
- Kaggle pour l’évaluation des modèles

---

## Livrables

- **GitHub** : Un dépôt contenant l’ensemble des codes, des modèles et un fichier descriptif pour chaque modèle.
- **Présentation** : Une présentation de 25 minutes couvrant la méthodologie et les résultats pour chaque objectif.

---

## Planification

- **Phase 1** : Prétraitement des données et sélection du modèle.
- **Phase 2** : Entraînement et validation du modèle.
- **Phase 3** : Correction des labels et explicabilité du modèle.
- **Phase 4** : Préparation des livrables et présentation.

---

## Conclusion

Ce projet requiert une approche structurée et méthodique pour atteindre les objectifs fixés. En suivant cette méthodologie, nous développerons un modèle de deep learning efficace pour l’identification des collemboles tout en assurant la correction des labels et l’explicabilité du modèle.

---


