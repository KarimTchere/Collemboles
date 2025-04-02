# -*- coding: utf-8 -*-
"""
Système d'identification des collemboles par deep learning avec architecture DynamicFPN
Conçu pour traiter efficacement des images haute résolution en préservant les
caractéristiques morphologiques essentielles à l'identification taxonomique.
"""

# Limiter à 8 cœurs CPU pour éviter la surcharge et optimiser les performances
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Éviter la fragmentation

# Optimisation de l'environnement
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["OMP_NUM_THREADS"] = "6"  # Utiliser vos 8 cœurs
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import gc
import torch
import atexit
import signal

# À ajouter au tout début du script, juste après les imports initiaux
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Non défini"))
print("Nombre de GPUs disponibles:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Mémoire totale: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")

# Optimisations avancées CUDA
# Priorité à la reproductibilité sur la vitesse
torch.backends.cudnn.benchmark = False  # Désactivé pour garantir des résultats identiques
torch.backends.cudnn.deterministic = True  # Activer le mode déterministe pour reproductibilité
if hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = True  # Activation des tenseurs float32
if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = True  # Accélération sur les GPU Ampere+
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('medium')  # Optimisation pour matrices de grande taille
torch.set_num_threads(6)  # Utilise tous vos cœurs pour PyTorch

def cleanup_resources():
    """Libère les ressources GPU et CPU avant la fin du programme"""
    print("\n[INFO] Nettoyage des ressources en cours...")
    
    # Libérer la mémoire GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                cached = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"GPU {i}: {allocated:.2f} GB alloués, {cached:.2f} GB en cache après nettoyage")
    
    # Forcer le garbage collector
    gc.collect()
    
    # Réinitialiser les variables d'environnement liées aux CPU
    os.environ["OMP_NUM_THREADS"] = ""
    os.environ["MKL_NUM_THREADS"] = ""
    os.environ["NUMEXPR_NUM_THREADS"] = ""
    
    print("[INFO] Ressources libérées. Au revoir!")

# Enregistrer la fonction pour qu'elle s'exécute à la fin du programme
atexit.register(cleanup_resources)

# Gérer aussi les interruptions (Ctrl+C)
def signal_handler(sig, frame):
    print("\n[INFO] Programme interrompu par l'utilisateur")
    # Sortir proprement appelle automatiquement cleanup_resources via atexit
    exit(0)

# Configurer les gestionnaires de signal
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command



import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import masks_to_boxes
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
from tqdm import tqdm
import random
import cv2
from collections import Counter
import warnings
import psutil
import math
from numpy import sqrt
from skimage.feature import hog as skimage_hog
from skimage import exposure
from contextlib import contextmanager
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import pickle
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration pour la reproductibilité complète des résultats
# Définir une graine fixe pour tous les générateurs de nombres aléatoires
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Configuration pour des opérations déterministes sur GPU
torch.backends.cudnn.deterministic = True  # Assure que les opérations CUDA sont déterministes
torch.backends.cudnn.benchmark = False     # Désactive l'optimisation auto qui peut causer de la variabilité

# Journal des paramètres de reproductibilité
print("="*50)
print("Configuration de la reproductibilité:")
print(f"- Graine aléatoire: {SEED}")
print(f"- CuDNN déterministe: {torch.backends.cudnn.deterministic}")
print(f"- CuDNN benchmark: {torch.backends.cudnn.benchmark}")
print("="*50)

# Définir les chemins des répertoires avec expansion du tilde
train_dir = os.path.expanduser("/home/barrage/ch_miashs_1_data/data_collemboles/data/data/")
test_dir = os.path.expanduser("/home/barrage/ch_miashs_1_data/data_collemboles/datatest/datatest/")

# Définir le dossier de sortie
output_dir = "/home/ch_miashs_1/Collemboles/soumissions/results/"
os.makedirs(output_dir, exist_ok=True)
print(f"Tous les résultats seront sauvegardés dans: {output_dir}")

# Constantes
N_EPOCHS = 30
BASE_COUNT = 200  # Minimum d'échantillons garantis pour chaque classe
INPUT_SIZE = 896   # Taille d'entrée pour les images (sur les dimensions les plus grandes)
batch_size = 4    

# Définir les constantes ImageNet pour la normalisation cohérente
IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_MEAN_PIXEL = tuple(int(x * 255) for x in IMAGENET_MEAN_RGB)  # (124, 116, 104)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)

# Mapping des classes
CLASS_MAPPING = {
    0: "AUTRE",
    1: "Cer",
    2: "CRY_THE",
    3: "HYP_MAN",
    4: "ISO_MIN",
    5: "LEP",
    6: "MET_AFF",
    7: "PAR_NOT",
    8: "FOND"
}

# Inverse du mapping pour la consultation par nom
CLASS_TO_IDX = {v: k for k, v in CLASS_MAPPING.items()}

# Mapping des projets (sera rempli dynamiquement)
PROJECT_MAPPING = {}


#=======================================================================================
# CLASSES ET FONCTIONS UTILITAIRES
#=======================================================================================

class EarlyStopping:
    """Arrête l'entraînement lorsque la métrique surveillée cesse de s'améliorer"""
    def __init__(self, patience=5, delta=0.001, monitor='loss'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.monitor = monitor  # 'loss' ou 'f1'

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
        elif (self.monitor == 'loss' and metric > self.best_score + self.delta) or \
             (self.monitor == 'f1' and metric < self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.counter = 0

def get_agreement_type(expert_labels):
    """Détermine le type d'accord entre les experts (4/4, 3/1, etc.)"""
    counts = Counter(expert_labels)
    max_count = max(counts.values()) if counts else 0
    unique_labels = len(counts)

    if max_count == 4:
        return "4/4"
    elif max_count == 3:
        return "3/1"
    elif max_count == 2 and unique_labels == 2:
        return "2/2"
    elif max_count == 2:
        return "2/1/1"
    else:
        return "1/1/1/1"

def expert_confidence_weight(expert_labels):
    """Calcule un poids de confiance basé sur l'accord entre experts"""
    counts = Counter(expert_labels)
    max_count = max(counts.values()) if counts else 0
    n_conflicts = sum(1 for v in counts.values() if v == max_count)

    # Cas 4/4
    if max_count == 4:
        return 1.0
    # Cas 3/1
    if max_count == 3:
        return 0.8
    # Cas 2/2
    if max_count == 2 and n_conflicts == 2:
        return 0.2  # Plus pénalisant que 2/1/1
    # Cas 2/1/1
    if max_count == 2:
        return 0.4
    # Cas 1/1/1/1
    return 0.1

def batch_confidence_weight(agreement_patterns):
    """Version vectorisée pour traiter les patterns d'accord par batch"""
    weights = {
        "4/4": 1.0,
        "3/1": 0.8,
        "2/2": 0.2,
        "2/1/1": 0.4,
        "1/1/1/1": 0.1
    }
    # Création sur CPU pour éviter les transferts inutiles
    return torch.tensor([weights.get(pattern, 0.1) for pattern in agreement_patterns])

def resolve_label_disagreement(expert_labels):
    """Résout les désaccords entre experts pour obtenir le label majoritaire"""
    vote_counts = {}
    for label in expert_labels:
        label_str = str(label)
        if label_str not in vote_counts:
            vote_counts[label_str] = 0
        vote_counts[label_str] += 1

    if not vote_counts:
        return 0  # Valeur par défaut si aucun label valide

    return int(max(vote_counts.items(), key=lambda x: x[1])[0])

class BalancedFocalLoss(nn.Module):
    def __init__(self, class_counts=None, gamma=2.0, alpha=0.25):
        super().__init__()

        # Exiger que class_counts soit fourni
        if class_counts is None:
            raise ValueError("class_counts doit être fourni pour BalancedFocalLoss")
        
        # CORRECTION: Standardisation cohérente du format pour exactement 9 classes
        if isinstance(class_counts, np.ndarray):
            # Ne pas appliquer np.bincount() si class_counts contient déjà les comptes
            # S'assurer d'avoir exactement 9 classes en tronquant ou complétant
            if len(class_counts) < 9:
                class_counts = np.pad(class_counts, (0, 9 - len(class_counts)), 'constant', constant_values=1)
            else:
                class_counts = class_counts[:9]  # Limiter aux 9 classes (0-8)
        elif isinstance(class_counts, list):
            if len(class_counts) < 9:
                class_counts.extend([1] * (9 - len(class_counts)))
            else:
                class_counts = class_counts[:9]  # Tronquer aux 9 premières classes
        elif isinstance(class_counts, dict):
            class_counts = [class_counts.get(i, 1) for i in range(9)]
        else:
            raise ValueError("Format de class_counts non supporté")

        # S'assurer que tous les comptages sont positifs pour éviter les erreurs
        counts_array = np.array(class_counts)
        counts_array = np.maximum(counts_array, 1)  # Éviter les divisions par zéro
        
        # Calcul des poids adaptatifs
        if np.max(counts_array) / np.min(counts_array) < 2.0:
            # Dataset déjà équilibré
            weights = 1.0 / (torch.sqrt(torch.tensor(counts_array, dtype=torch.float32) + 10.0))
        else:
            # Dataset déséquilibré
            weights = 1.0 / (torch.sqrt(torch.tensor(counts_array, dtype=torch.float32)) + 1e-5)

        # Traitement spécial pour la classe FOND (8)
        fond_count = counts_array[8]
        if fond_count > 100:
            weights[8] = weights[8] * 0.8  # Réduire si bien représenté
        elif fond_count < 20:
            weights[8] = weights[8] * 1.5  # Augmenter si rare

        # Normaliser les poids
        self.weights = weights * (len(weights) / weights.sum())
        self.gamma = gamma
        self.alpha = alpha
        
        # Logging des poids
        print("Poids des classes pour Focal Loss:")
        for i, w in enumerate(self.weights):
            print(f"  Classe {i} ({CLASS_MAPPING.get(i, 'Inconnue')}): {w.item():.4f}")
    
    def forward(self, inputs, targets):
        device = inputs.device
        # Déplacer les poids vers le device des inputs (GPU)
        weights = self.weights.to(device)  # Créer une copie sur GPU
        
        # Amélioration de la stabilité numérique
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Écrêtage pour éviter les instabilités numériques
        BCE_loss = torch.clamp(BCE_loss, min=0, max=50)
        
        # Limitation de l'exponentielle pour éviter les débordements
        pt = torch.clamp(torch.exp(-BCE_loss), min=1e-7, max=1-1e-7)
        focal_weight = self.alpha * (1-pt)**self.gamma

        # Utiliser les poids qui sont maintenant sur le même device que targets
        class_weight = weights[targets]
        
        # Remplacer les NaN et Inf par des valeurs sûres
        loss = class_weight.detach() * focal_weight.detach() * BCE_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=0.0)

        return loss.mean()

# Fonction utilitaire pour surveiller la mémoire GPU
def print_gpu_memory(step_name=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Memory [{step_name}]: {allocated:.2f}/{max_mem:.2f} GB allocated, {reserved:.2f} GB reserved")

def clear_gpu_memory(verbose=False, force_full_cleanup=False):
    """
    Libère la mémoire GPU de manière optimisée et intelligente.
    
    Args:
        verbose (bool): Si True, affiche des informations détaillées
        force_full_cleanup (bool): Si True, effectue un nettoyage plus agressif
                                  (à utiliser uniquement entre les phases importantes)
    """
    if not torch.cuda.is_available():
        return
        
    # Mesurer la mémoire avant nettoyage
    mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
    
    # Nettoyage standard - suffisant dans la plupart des cas
    torch.cuda.empty_cache()
    gc.collect()
    
    # Nettoyage agressif uniquement si demandé explicitement
    if force_full_cleanup:
        torch.cuda.synchronize()  # Coûteux en performance
        for _ in range(2):
            gc.collect()
            torch.cuda.empty_cache()
    
    # Mesurer la mémoire après nettoyage
    if verbose:
        mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        # Afficher uniquement si la différence est significative ou si verbose
        if mem_before - mem_after > 0.1 or verbose:
            print(f"Mémoire GPU: {mem_after:.3f} GB alloués, {mem_reserved:.3f} GB réservés "
                  f"(libéré: {mem_before - mem_after:.3f} GB)")

def debug_tensor_shapes(batch_idx, inputs, metadata):
    """Affiche des informations détaillées sur les formes des tenseurs pour le débogage"""
    print(f"\n=== Débogage du batch {batch_idx} ===")
    print(f"- inputs: shape={inputs.shape}, dtype={inputs.dtype}, device={inputs.device}")
    # Adapter le message pour prendre en compte le nombre variable de canaux
    num_channels = inputs.size(1) if inputs.dim() > 3 else inputs.size(0)
    print(f"- Nombre de canaux: {num_channels} (RGB + HOG)")
    print(f"- inputs[0]: min={inputs[0].min().item():.4f}, max={inputs[0].max().item():.4f}")

    if 'padding_mask' in metadata:
        mask = metadata['padding_mask']
        if isinstance(mask, torch.Tensor):
            print(f"- padding_mask: shape={mask.shape}, dtype={mask.dtype}, device={mask.device}")
            if mask.numel() > 0:
                print(f"  values: min={mask.min().item():.1f}, max={mask.max().item():.1f}, "
                      f"non-zeros={torch.count_nonzero(mask).item()}/{mask.numel()}")
        else:
            print(f"- padding_mask: type={type(mask)}")
    
    if 'size' in metadata:
        size = metadata['size']
        if isinstance(size, torch.Tensor):
            print(f"- size: shape={size.shape}, dtype={size.dtype}, device={size.device}")
            if size.numel() > 0 and size.dim() > 0:
                print(f"  first element: {size[0].tolist()}")
        else:
            print(f"- size: type={type(size)}")

def debug_memory_usage(prefix="", top_n=5):
    """
    Identifie les plus grands tenseurs en mémoire GPU pour diagnostiquer les fuites.
    
    Args:
        prefix (str): Préfixe pour le message (étape actuelle)
        top_n (int): Nombre de grands tenseurs à afficher
    """
    if not torch.cuda.is_available():
        return
        
    print(f"\n=== Analyse détaillée mémoire GPU {prefix} ===")
    
    # Forcer un garbage collection
    gc.collect()
    torch.cuda.synchronize()
    
    # Informations générales
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"Mémoire allouée: {allocated:.3f} GB ({allocated/max_mem*100:.1f}% du total)")
    print(f"Mémoire réservée: {reserved:.3f} GB ({reserved/max_mem*100:.1f}% du total)")
    print(f"Cache mémoire: {reserved-allocated:.3f} GB")
    
    # Collecter et trier les tenseurs par taille
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'cuda':
                size_bytes = obj.element_size() * obj.nelement()
                tensors.append((obj.shape, size_bytes, obj.dtype))
        except Exception:
            pass
            
    if tensors:
        tensors.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop {min(top_n, len(tensors))} tenseurs GPU par taille:")
        for i, (shape, size, dtype) in enumerate(tensors[:top_n]):
            print(f"{i+1}. Forme: {shape}, Type: {dtype}, Taille: {size/(1024**2):.2f} MB")
        
        # Statistiques sur les types de tenseurs
        dtype_sizes = {}
        for _, size, dtype in tensors:
            if dtype not in dtype_sizes:
                dtype_sizes[dtype] = 0
            dtype_sizes[dtype] += size
        
        print("\nUtilisation mémoire par type de données:")
        for dtype, size in sorted(dtype_sizes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {dtype}: {size/(1024**3):.3f} GB ({size/(1024**2):.1f} MB)")
    else:
        print("Aucun tenseur GPU trouvé.")
    
    # Vérification de la fragmentation (si disponible)
    try:
        stats = torch.cuda.memory_stats()
        if 'allocated_bytes.all.current' in stats:
            active = stats['active_bytes.all.current'] / (1024**3)
            print(f"\nMémoire active: {active:.3f} GB")
            if 'reserved_bytes.all.current' in stats:
                fragmentation = 1.0 - (stats['active_bytes.all.current'] / stats['reserved_bytes.all.current'])
                print(f"Fragmentation: {fragmentation:.1%}")
    except Exception as e:
        print(f"Statistiques avancées non disponibles: {e}")
    
    print("=" * 50)

def transform_bbox(bbox, angle, image_size):
    """
    Calcule une nouvelle bounding box alignée avec les axes qui englobe complètement 
    la bbox originale après rotation.
    
    Args:
        bbox: Tuple (x1, y1, x2, y2) de la bounding box originale
        angle: Angle de rotation en degrés
        image_size: Tuple (width, height) de l'image originale
    
    Returns:
        Tuple (x1, y1, x2, y2) de la nouvelle bounding box englobante
    """
    x1, y1, x2, y2 = bbox
    img_width, img_height = image_size
    
    # Calculer le centre de l'image (point de pivot pour la rotation)
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    
    # Convertir l'angle en radians
    angle_rad = math.radians(angle)
    
    # Calculer les coordonnées des quatre coins de la bbox originale
    corners = [
        (x1, y1),  # Coin supérieur gauche
        (x2, y1),  # Coin supérieur droit
        (x2, y2),  # Coin inférieur droit
        (x1, y2)   # Coin inférieur gauche
    ]
    
    # Rotation des coins autour du centre de l'image
    rotated_corners = []
    for x, y in corners:
        # Coordonnées relatives au centre de l'image
        x_centered = x - img_center_x
        y_centered = y - img_center_y
        
        # Appliquer la rotation
        x_rotated = x_centered * math.cos(angle_rad) - y_centered * math.sin(angle_rad)
        y_rotated = x_centered * math.sin(angle_rad) + y_centered * math.cos(angle_rad)
        
        # Retour aux coordonnées absolues
        x_absolute = x_rotated + img_center_x
        y_absolute = y_rotated + img_center_y
        
        rotated_corners.append((x_absolute, y_absolute))
    
    # Déterminer les limites du nouveau rectangle englobant aligné sur les axes
    min_x = max(0, min(corner[0] for corner in rotated_corners))
    min_y = max(0, min(corner[1] for corner in rotated_corners))
    max_x = min(img_width, max(corner[0] for corner in rotated_corners))
    max_y = min(img_height, max(corner[1] for corner in rotated_corners))
    
    # S'assurer que les dimensions sont valides
    if min_x >= max_x or min_y >= max_y:
        return bbox  # Revenir à la bbox originale en cas de problème
    
    return (int(min_x), int(min_y), int(max_x), int(max_y))

#=======================================================================================
# ARCHITECTURE DYNAMICFPN
#=======================================================================================

class DynamicFPN(nn.Module):
    """
    Architecture Feature Pyramid Network adaptée aux images haute résolution
    avec traitement multi-échelle et préservation des caractéristiques taxonomiques
    """
    def __init__(self, num_classes=9, backbone_name='resnet50', pretrained=True):
        super().__init__()
        
        # Choix du backbone
        if backbone_name == 'resnet50':
            # Forcer l'initialisation sur CPU et désactiver temporairement les gradients
            with torch.no_grad():
                # Charger le backbone
                self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None).cpu()
                
                # Sauvegarder les poids originaux
                original_weights = self.backbone.conv1.weight.clone().cpu()
                
                # Créer nouvelle couche conv1
                new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False).cpu()
                
                # Initialisation des canaux RGB - copie standard
                new_conv1.weight.data[:, :3, :, :] = original_weights[:, :3, :, :]
                
                # Initialisation adaptée aux caractéristiques HOG (qui sont des gradients normalisés)
                # Utiliser une distribution normale avec une variance plus faible
                nn.init.normal_(new_conv1.weight.data[:, 3:4, :, :], 
                                mean=0.0, 
                                std=0.01)  # Variance réduite pour stabiliser l'apprentissage initial
                
                # Remplacer la couche originale
                self.backbone.conv1 = new_conv1
                
                # Libérer la mémoire
                del original_weights, new_conv1
                gc.collect()
                
            
            self.feature_channels = [256, 512, 1024, 2048]
            
            # Réduire la complexité du réseau FPN
            # self.fpn_channels = 128  # pour économiser de la mémoire

        elif backbone_name == 'resnext50':
            self.backbone = models.resnext50_32x4d(weights='IMAGENET1K_V1' if pretrained else None)
            
            # Même processus pour ResNeXt
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3, :, :] = original_conv1.weight.clone()
                self.backbone.conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
            
            self.feature_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Backbone {backbone_name} non supporté. Utilisez 'resnet50' ou 'resnext50'")
        # Paramètres FPN
        self.fpn_channels = 256
        self.num_scales = 4

        # Couches latérales FPN pour réduire la dimensionnalité
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, self.fpn_channels, kernel_size=1)
            for channels in self.feature_channels
        ])

        # Convolutions de fusion 3x3 pour améliorer les features
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
            for _ in range(self.num_scales)
        ])

        # Attention spatiale pour mettre en évidence les caractéristiques taxonomiques importantes
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fpn_channels, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            ) for _ in range(self.num_scales)
        ])

        # Traitement des métadonnées de taille - crucial pour les caractéristiques morphologiques
        self.size_encoder = nn.Sequential(
            nn.Linear(3, 64),  # Largeur, hauteur, ratio
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Pooling adaptatif pour gérer les images de tailles variables
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classification finale avec dropout pour éviter le surapprentissage
        self.classifier = nn.Sequential(
            nn.Linear(self.fpn_channels * self.num_scales + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

        # Tête d'incertitude pour estimer la confiance dans la prédiction
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.fpn_channels * self.num_scales + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, sizes=None, padding_mask=None):
        """
        Forward pass avec support des métadonnées de taille et masques de padding
        """
        # Validation dimensionnelle (inchangée)
        if x.dim() > 4:
            print(f"CORRECTION dans forward: Entrée avec dimensions incorrectes: {x.shape}")
            b = x.size(0)
            x = x.view(b, 4, x.size(-2), x.size(-1))
            print(f"Corrigé en: {x.shape}")
        elif x.dim() != 4:
            raise ValueError(f"Le tenseur d'entrée doit être 4D [batch, channels, height, width], mais a forme {x.shape}")
        
        assert x.size(1) == 4, f"L'entrée doit avoir 4 canaux (RGB+HOG), mais a {x.size(1)} canaux"

        # Traitement du masque de padding (inchangé)
        if padding_mask is not None:
            if padding_mask.dim() > 3:
                b = padding_mask.size(0)
                padding_mask = padding_mask.view(b, 1, padding_mask.size(-2), padding_mask.size(-1))
            
            if padding_mask.shape[-2:] != x.shape[-2:]:
                mask_resized = F.interpolate(padding_mask.float(), size=x.shape[2:], mode='nearest')
                x = x * mask_resized
            else:
                x = x * padding_mask

        # Extraction des features à différentes résolutions
        features = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # CORRECTION: Extraction par couches SANS gradient checkpointing
        x1 = self.backbone.layer1(x)  # Retiré checkpoint()
        features.append(x1)
        
        x2 = self.backbone.layer2(x1)  # Retiré checkpoint()
        features.append(x2)
        
        x3 = self.backbone.layer3(x2)  # Retiré checkpoint()
        features.append(x3)
        
        x4 = self.backbone.layer4(x3)  # Retiré checkpoint()
        features.append(x4)

        # Appliquer le masque à chaque niveau de feature
        if padding_mask is not None:
            for i in range(len(features)):
                feature_mask = F.interpolate(padding_mask.float(), 
                                            size=features[i].shape[2:], 
                                            mode='nearest')
                features[i] = features[i] * feature_mask

        # Construction FPN (réduction de dimensionnalité)
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Fusion top-down (propager l'information sémantique vers les niveaux détaillés)
        merged = []
        for i in range(len(laterals)-1, -1, -1):
            if i == len(laterals)-1:
                merged.append(laterals[i])
            else:
                # Upsampling de la feature de niveau supérieur et addition
                up = F.interpolate(merged[-1], size=laterals[i].shape[2:], mode='nearest')
                merged.append(laterals[i] + up)

        merged = merged[::-1]  # Inverser l'ordre (P2, P3, P4, P5)

        # Application des convolutions de fusion et attention spatiale
        fpn_outputs = []
        for i, (conv, m, att) in enumerate(zip(self.fusion_convs, merged, self.spatial_attention)):
            feat = conv(m)  # Affiner les features
            attention_map = att(feat)  # Générer la carte d'attention
            fpn_outputs.append(feat * attention_map)  # Multiplier les features par l'attention
        
        # Agrégation spatiale
        pooled = [self.avgpool(f).flatten(1) for f in fpn_outputs]
        visual_features = torch.cat(pooled, dim=1)
        
        # Ajout des métadonnées de taille si disponibles
        if sizes is not None:
            size_features = self.size_encoder(sizes)
            combined = torch.cat([visual_features, size_features], dim=1)
        else:
            combined = visual_features

        # Classification et incertitude
        logits = self.classifier(combined)
        uncertainty = self.uncertainty_head(combined)

        return logits, uncertainty

    def extract_features(self, x, sizes=None, padding_mask=None):
        """
        Extrait les features sans classification finale pour l'approche par patchs
        Optimisé pour la gestion CPU/GPU et les problèmes de mémoire
        """
        # Vérifier et corriger les dimensions d'entrée si nécessaire
        if x.dim() > 4:
            print(f"CORRECTION dans extract_features: Entrée avec dimensions incorrectes: {x.shape}")
            b = x.size(0)
            x = x.view(b, 4, x.size(-2), x.size(-1))  # Correction pour 4 canaux (RGB+HOG)
            print(f"Corrigé en: {x.shape}")
        
        # Vérifier que nous avons bien 4 canaux (RGB+HOG)
        if x.size(1) != 4:
            print(f"ATTENTION: L'entrée a {x.size(1)} canaux au lieu de 4 (RGB+HOG)")
            # Si moins de 4 canaux, ajouter un canal de zéros
            if x.size(1) < 4:
                zeros = torch.zeros((x.size(0), 4-x.size(1), x.size(2), x.size(3)), 
                                dtype=x.dtype, device=x.device)
                x = torch.cat([x, zeros], dim=1)
            # Si plus de 4 canaux, tronquer
            else:
                x = x[:, :4, :, :]
            print(f"Corrigé en: {x.shape}")
        
        # Appliquer le masque de padding si fourni
        if padding_mask is not None:
            # Vérifier et corriger les dimensions du masque
            if padding_mask.dim() > 3:
                print(f"CORRECTION dans extract_features: Masque avec dimensions incorrectes: {padding_mask.shape}")
                b = padding_mask.size(0)
                padding_mask = padding_mask.view(b, 1, padding_mask.size(-2), padding_mask.size(-1))
                print(f"Corrigé en: {padding_mask.shape}")
            
            # Application correcte du masque sans ajouter de dimension
            if padding_mask.shape[-2:] == x.shape[-2:]:
                x = x * padding_mask  # Ne pas utiliser unsqueeze ici
            else:
                print(f"Attention: Les dimensions du masque {padding_mask.shape[-2:]} ne correspondent pas à l'entrée {x.shape[-2:]}")
        
        # Fonction pour libérer la mémoire GPU pendant le calcul
        def clear_intermediates():
            # CORRECTION : Vérifier directement la disponibilité de CUDA au lieu de dépendre de x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Extraction des features par étapes pour économiser la mémoire
        with torch.cuda.amp.autocast():  # Utiliser la précision mixte même en inférence
            # Étape 1: Features initiales
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            # Étape 2: Extraire les features par couche, libérer mémoire entre chaque
            features = []
            
            # Layer 1
            x1 = self.backbone.layer1(x)
            features.append(x1)
            del x  # Libérer la mémoire dès que possible
            clear_intermediates()
            
            # Layer 2
            x2 = self.backbone.layer2(x1)
            features.append(x2)
            del x1
            clear_intermediates()
            
            # Layer 3
            x3 = self.backbone.layer3(x2)
            features.append(x3)
            del x2
            clear_intermediates()
            
            # Layer 4
            x4 = self.backbone.layer4(x3)
            features.append(x4)
            del x3
            clear_intermediates()
            
            # Étape 3: Réduction de dimensionnalité via les convolutions latérales
            laterals = []
            for i, (conv, f) in enumerate(zip(self.lateral_convs, features)):
                lat = conv(f)
                laterals.append(lat)
                # Libérer les features originales après traitement
                features[i] = None  # Pour aider le GC
            
            clear_intermediates()
            
            # Étape 4: Fusion top-down
            merged = []
            for i in range(len(laterals)-1, -1, -1):
                if i == len(laterals)-1:
                    merged.append(laterals[i])
                else:
                    up = F.interpolate(merged[-1], size=laterals[i].shape[2:], mode='nearest')
                    merged.append(laterals[i] + up)
                    # Libérer la mémoire des latéraux après fusion
                    laterals[i] = None
            
            merged = merged[::-1]  # Inverser l'ordre
            clear_intermediates()
            
            # Étape 5: Application des convolutions de fusion et attention
            fpn_outputs = []
            for i, (conv, m, att) in enumerate(zip(self.fusion_convs, merged, self.spatial_attention)):
                feat = conv(m)
                attention_map = att(feat)
                fpn_outputs.append(feat * attention_map)
                # Libérer la mémoire
                merged[i] = None
                
            clear_intermediates()
            
            # Étape 6: Agrégation spatiale
            pooled = []
            for i, f in enumerate(fpn_outputs):
                p = self.avgpool(f).flatten(1)
                pooled.append(p)
                # Libérer la mémoire
                fpn_outputs[i] = None
                
            del fpn_outputs
            visual_features = torch.cat(pooled, dim=1)
            del pooled
            clear_intermediates()

            # Nettoyage final explicite de la mémoire GPU
            if torch.cuda.is_available():
                # Forcer une synchronisation et un nettoyage complet
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
            
            # Étape 7: Combinaison avec les caractéristiques de taille
            if sizes is not None:
                size_features = self.size_encoder(sizes)
                combined = torch.cat([visual_features, size_features], dim=1)
            else:
                combined = visual_features
                del visual_features
        
        return combined

    def classify_features(self, features):
        """Classifie des features pré-extraites pour l'approche par patchs"""
        logits = self.classifier(features)
        uncertainty = self.uncertainty_head(features)
        return logits, uncertainty

#=======================================================================================
# TRANSFORMATIONS ET PRÉTRAITEMENT D'IMAGES CORRIGÉ
#=======================================================================================

def normalize_tensor(tensor):
    """Normalise un tenseur d'image selon les statistiques ImageNet"""
    if tensor.size(0) == 3:
        # Cas standard: 3 canaux RGB
        return TF.normalize(tensor, IMAGENET_MEAN_RGB, IMAGENET_STD_RGB)
    elif tensor.size(0) == 4:
        # Cas spécial: 4 canaux (RGB + HOG)
        # Normaliser uniquement les 3 premiers canaux (RGB)
        rgb_channels = tensor[:3]
        hog_channel = tensor[3:4]  # Garde la dimension du canal
        
        normalized_rgb = TF.normalize(rgb_channels, IMAGENET_MEAN_RGB, IMAGENET_STD_RGB)
        return torch.cat([normalized_rgb, hog_channel], dim=0)
    else:
        # Gestion des autres cas avec message d'avertissement
        print(f"Attention: Le tenseur a {tensor.size(0)} canaux, 3 ou 4 attendus. "
              f"Tentative de normalisation des 3 premiers canaux.")
        
        if tensor.size(0) > 3:
            # Si plus de 3 canaux, normaliser les 3 premiers et préserver les autres
            rgb_channels = tensor[:3]
            other_channels = tensor[3:]
            
            normalized_rgb = TF.normalize(rgb_channels, IMAGENET_MEAN_RGB, IMAGENET_STD_RGB)
            return torch.cat([normalized_rgb, other_channels], dim=0)
        else:
            # Pour les tenseurs avec moins de 3 canaux, normaliser ce que nous avons
            return TF.normalize(tensor, 
                              IMAGENET_MEAN_RGB[:tensor.size(0)], 
                              IMAGENET_STD_RGB[:tensor.size(0)])

class PreserveAspectRatioPad:
    """
    Transforme les images en préservant le ratio d'aspect et SANS redimensionner les grandes images.
    Ajoute uniquement un padding pour les images sous-dimensionnées.
    """
    def __init__(self, target_size=896):
        self.target_size = target_size

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            return img

        # Ne pas redimensionner si l'image est déjà suffisamment grande
        if img.width >= self.target_size and img.height >= self.target_size:
            return img  # Conserver la résolution originale pour les grandes images
            
        # Calculer le ratio pour atteindre la taille cible UNIQUEMENT si nécessaire
        ratio = min(self.target_size / img.width, self.target_size / img.height)
        
        # Redimensionner en préservant le ratio d'aspect seulement si l'image est trop petite
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Si les dimensions sont égales à la cible, pas besoin de padding
        if new_w == self.target_size and new_h == self.target_size:
            return img
            
        # Créer une image avec padding utilisant la moyenne ImageNet comme couleur de fond
        padded = Image.new("RGB", (self.target_size, self.target_size), IMAGENET_MEAN_PIXEL)
        
        # Centrer l'image
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        padded.paste(img, (x_offset, y_offset))
        
        return padded

class SmartResizeWithMetadata:
    """
    Stratégie unifiée et cohérente pour le redimensionnement des images
    qui respecte les caractéristiques morphologiques et taxonomiques.
    """
    def __init__(self, target_size=INPUT_SIZE, preserve_large=True, resize_mode="lanczos"):
        self.target_size = target_size
        self.preserve_large = preserve_large  # Si True, ne downscale pas les grandes images
        self.resize_mode = resize_mode
        self.resize_method = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST
        }.get(resize_mode.lower(), Image.LANCZOS)
        
    def __repr__(self):
        """
        Représentation lisible de l'objet pour le débogage et la traçabilité.
        Facilite l'identification des paramètres utilisés lors du traitement.
        """
        return f"SmartResize(target={self.target_size}, preserve_large={self.preserve_large}, mode={self.resize_mode})"
        
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            return img, {}  # CORRIGÉ: Retourne toujours un tuple avec dict vide
            
        w, h = img.size
        size_info = {'original_size': (w, h)}
        
        if max(w, h) <= self.target_size and not self.preserve_large:
            # Redimensionnement si nécessaire
            ratio = min(self.target_size / w, self.target_size / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), self.resize_method)
            size_info['resize_operation'] = 'downscaled'
            size_info['resize_ratio'] = ratio
        elif max(w, h) < self.target_size:
            # Upscaling uniquement si plus petit que la cible
            ratio = max(self.target_size / w, self.target_size / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), self.resize_method)
            size_info['resize_operation'] = 'upscaled'
            size_info['resize_ratio'] = ratio
        else:
            size_info['resize_operation'] = 'preserved'
            size_info['resize_ratio'] = 1.0
        
        # Créer l'image padée si nécessaire
        new_w, new_h = img.size
        if new_w == self.target_size and new_h == self.target_size:
            # Déjà aux dimensions cibles
            size_info['padding_added'] = False
            return img, size_info  # Retourne toujours un tuple
        
        padded = Image.new("RGB", (self.target_size, self.target_size), IMAGENET_MEAN_PIXEL)
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        padded.paste(img, (x_offset, y_offset))
        
        size_info['padding_added'] = True
        size_info['padding_offsets'] = (x_offset, y_offset)
        return padded, size_info  # Retourne un tuple
    
class GaussianNoise(object):
    """
    Ajoute du bruit gaussien à une image.
    Particulièrement utile pour les images microscopiques de collemboles.
    
    Args:
        mean (float): Moyenne du bruit gaussien (généralement 0)
        std (float, tuple): Écart-type du bruit ou plage d'écarts-types (min, max)
        p (float): Probabilité d'appliquer la transformation
    """
    def __init__(self, mean=0., std=(0.01, 0.05), p=0.5):
        self.mean = mean
        self.std = std
        self.p = p
        
    def __call__(self, image):
        if random.random() < self.p:
            # Pour les images PIL
            if isinstance(image, Image.Image):
                image = np.array(image).astype(np.float32) / 255.0
                
                # Choisir un écart-type aléatoire dans la plage si une plage est spécifiée
                std = self.std if isinstance(self.std, (int, float)) else random.uniform(*self.std)
                
                # Générer et appliquer le bruit
                noise = np.random.normal(self.mean, std, image.shape)
                noisy_image = image + noise
                
                # Ajustement pour rester dans la plage [0, 1]
                noisy_image = np.clip(noisy_image, 0, 1)
                
                # Reconvertir en image PIL
                return Image.fromarray((noisy_image * 255).astype(np.uint8))
            
            # Pour les tenseurs
            elif isinstance(image, torch.Tensor):
                # Choisir un écart-type aléatoire dans la plage si une plage est spécifiée
                std = self.std if isinstance(self.std, (int, float)) else random.uniform(*self.std)
                
                # Ne générer du bruit que pour les canaux RGB (3 premiers canaux)
                if image.dim() == 3 and image.size(0) >= 3:
                    # Créer un tenseur de bruit seulement pour les canaux RGB
                    noise_rgb = torch.randn(3, *image.shape[1:]) * std + self.mean
                    
                    # Appliquer le bruit seulement aux canaux RGB
                    noisy_image = image.clone()
                    noisy_image[:3] = noisy_image[:3] + noise_rgb
                    
                    # Laisser le canal HOG inchangé s'il existe
                    # Ajustement pour rester dans la plage normale
                    noisy_image[:3] = torch.clamp(noisy_image[:3], 0, 1)
                    return noisy_image
                else:
                    # Comportement standard pour les autres cas
                    noise = torch.randn_like(image) * std + self.mean
                    noisy_image = image + noise
                    return torch.clamp(noisy_image, 0, 1)
        
        # Pas de modification si la transformation n'est pas appliquée
        return image
    
def compute_hog(image, target_size=None):
    """
    Calcule les caractéristiques HOG pour capturer la structure morphologique
    Version standardisée utilisant scikit-image pour une meilleure fiabilité
    
    Args:
        image: Image PIL ou array numpy
        target_size: Tuple (width, height) pour le redimensionnement final, ou None
    
    Returns:
        tuple: Descripteurs HOG et image de visualisation HOG
    """
    # Conversion en niveaux de gris adaptée au type d'entrée
    if isinstance(image, Image.Image):
        if target_size:
            # Redimensionner d'abord si une taille cible est spécifiée
            image = image.resize(target_size, Image.LANCZOS)
        img_array = np.array(image.convert('L'))
    elif isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
        # Convertir RGB en niveaux de gris avec méthode optimisée
        img_array = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        if target_size:
            img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
    else:
        img_array = image  # Déjà en niveaux de gris
        if target_size:
            img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
    
    # Assurer que l'image est 2D
    if img_array.ndim > 2:
        img_array = img_array[:,:,0]  # Prendre seulement le premier canal
    
    # Amélioration du contraste pour capturer plus de détails morphologiques
    p2, p98 = np.percentile(img_array, (2, 98))
    img_array = np.clip(img_array, p2, p98)
    img_array = (img_array - p2) / (p98 - p2 + 1e-8) * 255
    
    # Utiliser l'implémentation standard de HOG avec gestion d'erreurs
    try:
        # Calcul HOG standard avec scikit-image
        fd, hog_visualization = skimage_hog(
            img_array, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            visualize=True,
            block_norm='L2-Hys'
        )
        
        # Normalisation pour la visualisation
        hog_visualization = exposure.rescale_intensity(hog_visualization, in_range=(0, 10))
        hog_visualization = (hog_visualization * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Erreur dans le calcul HOG standard: {e}. Utilisation d'une alternative.")
        # Version de secours - image de gradient simple
        try:
            gx = cv2.Sobel(img_array.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img_array.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)
            hog_visualization = cv2.GaussianBlur(magnitude, (0, 0), 0.5).astype(np.uint8)
        except Exception as e2:
            print(f"Erreur de repli HOG: {e2}, création d'un canal de zéros")
            # En dernier recours, utiliser un canal de zéros
            h, w = img_array.shape
            hog_visualization = np.zeros((h, w), dtype=np.uint8)
    
    # Vérification finale de la forme
    if target_size and hog_visualization.shape != (target_size[1], target_size[0]):
        hog_visualization = cv2.resize(hog_visualization, target_size, interpolation=cv2.INTER_LINEAR)
    
    return fd, hog_visualization.astype(np.uint8)


# Dans la fonction dynamic_transform - Modification de l'approche
def dynamic_transform(image, add_metadata=False, precomputed_hog=None):
    """
    Fonction unifiée pour transformer les images en tenseurs et ajouter les métadonnées.
    Calcule systématiquement le HOG après les transformations.
    """
    if isinstance(image, torch.Tensor):
        # Déjà un tenseur
        return image
        
    if not isinstance(image, Image.Image):
        # Convertir en image PIL si nécessaire (np.array, etc.)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            return image  # Impossible de traiter ce type de données
    
    # Étape 1: Redimensionner l'image sans padding
    original_size = image.size
    
    # Calculer le ratio pour préserver les proportions
    ratio = min(INPUT_SIZE / max(original_size), 1.0 if max(original_size) > INPUT_SIZE else 10.0)
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    resized_img = image.resize(new_size, Image.LANCZOS)
    
    # Étape 2: Calculer systématiquement le HOG sur l'image redimensionnée (après augmentation)
    # Nous ignorons le precomputed_hog car nous voulons toujours recalculer le HOG
    try:
        # Convertir en niveaux de gris pour HOG
        gray_img = resized_img.convert('L')
        
        # Calculer les descripteurs HOG et la visualisation
        _, hog_visualization = compute_hog(gray_img)
        
        # Vérifier que les dimensions correspondent
        if hog_visualization.shape != (new_size[1], new_size[0]):
            # Redimensionner la visualisation HOG si nécessaire
            hog_visualization = cv2.resize(hog_visualization, new_size, interpolation=cv2.INTER_AREA)
        
        # Convertir la visualisation HOG en tenseur (canal unique)
        hog_tensor = torch.from_numpy(hog_visualization).float().unsqueeze(0) / 255.0
    except Exception as e:
        print(f"Erreur HOG: {e}")
        # Créer un canal HOG basé sur les bords (détection de contours simple)
        try:
            # Utiliser une détection de contours Canny comme repli
            gray_array = np.array(resized_img.convert('L'))
            edges = cv2.Canny(gray_array, 50, 150)
            hog_tensor = torch.from_numpy(edges).float().unsqueeze(0) / 255.0
        except Exception as e2:
            print(f"Erreur de repli HOG: {e2}, création d'un canal de zéros")
            # En dernier recours, utiliser un canal de zéros
            hog_tensor = torch.zeros((1, new_size[1], new_size[0]), dtype=torch.float32)
    
    # Étape 3: Appliquer le padding à l'image redimensionnée
    padded_img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), IMAGENET_MEAN_PIXEL)
    x_offset = (INPUT_SIZE - new_size[0]) // 2
    y_offset = (INPUT_SIZE - new_size[1]) // 2
    padded_img.paste(resized_img, (x_offset, y_offset))
    
    # Convertir en tenseur et normaliser selon les statistiques ImageNet
    tensor = TF.to_tensor(padded_img)
    normalized = normalize_tensor(tensor)
    
    # Créer les métadonnées
    metadata = {
        'original_size': original_size,
        'resized_size': new_size,
        'padding_offsets': (x_offset, y_offset),
        'padding_added': True,
        'resize_ratio': ratio
    }
    
    # Étape 4: Appliquer le padding au canal HOG ou créer un canal fictif
    if hog_tensor is not None:
        # Créer un tenseur HOG padé
        padded_hog = torch.zeros((1, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)
        padded_hog[:, y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = hog_tensor
    else:
        # Canal HOG factice
        padded_hog = torch.zeros((1, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)
    
    # Fusion: Ajouter HOG comme 4ème canal
    enhanced = torch.cat([normalized, padded_hog], dim=0)
    assert enhanced.size(0) == 4, f"Le tenseur doit avoir exactement 4 canaux (RGB+HOG), mais a {enhanced.size(0)} canaux"
    
    if add_metadata:
        return enhanced, metadata
    return enhanced

def get_collembole_transforms(training=False, target_size=INPUT_SIZE, with_noise=True):
    """
    Transformations adaptées aux caractéristiques morphologiques des collemboles.
    Version modifiée qui assure que HOG est calculé APRÈS augmentation.
    """
    transform_list = []
    
    if training:
        # Appliquer toutes les transformations géométriques et de couleur d'abord
        transform_list.extend([
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            GaussianNoise(mean=0, std=(0.01, 0.03), p=0.7) if with_noise else transforms.Lambda(lambda x: x),
        ])
    
    # Étape finale: calcul du HOG après toutes les augmentations
    transform_list.append(transforms.Lambda(lambda img: dynamic_transform(img, add_metadata=True)))
        
    return transforms.Compose(transform_list)

def create_padding_mask(image_size, target_size=INPUT_SIZE):
    """
    Crée un masque binaire pour identifier les pixels réels vs padding.
    Le masque a des 1 pour le contenu réel de l'image et des 0 pour le padding.
    Optimisé pour la cohérence et la gestion des cas limites.
    """
    w, h = image_size
    
    # Vérification des dimensions
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
        print(f"Dimensions invalides: {image_size}, utilisation de valeurs par défaut")
        w, h = 1, 1
    
    # S'assurer que les dimensions sont positives
    w, h = max(1, w), max(1, h)
    max_dim = max(w, h)
    
    # Si l'image est plus grande que la taille cible, retourner un masque plein
    if max_dim >= target_size:
        return torch.ones((1, target_size, target_size))
        
    # Calculer le ratio de redimensionnement avec vérification de sécurité
    ratio = min(target_size / max_dim, 10.0)  # Limiter le ratio pour éviter des dimensions extrêmes
    new_w, new_h = int(w * ratio), int(h * ratio)
    
    # S'assurer que les nouvelles dimensions sont valides
    new_w, new_h = min(target_size, max(1, new_w)), min(target_size, max(1, new_h))
    
    # Calculer les offsets de padding
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # Créer le masque avec des 1 pour la zone d'image réelle (sur CPU)
    mask = torch.zeros((1, target_size, target_size))
    mask[:, y_offset:y_offset+new_h, x_offset:x_offset+new_w] = 1
    
    # Vérification de validité
    assert mask.sum() > 0, "Le masque ne doit pas être entièrement vide"
    assert new_w > 0 and new_h > 0, f"Dimensions calculées invalides: {new_w}x{new_h}"
    
    return mask

def validate_preprocessing_pipeline(image_path, target_size=INPUT_SIZE):
    """
    Valide le pipeline complet de prétraitement en affichant des diagnostics
    à chaque étape. Outil essentiel pour déboguer les problèmes de prétraitement.
    
    Args:
        image_path: Chemin vers une image de test
        target_size: Taille cible pour le prétraitement
    """
    print(f"Validation du pipeline de prétraitement pour: {image_path}")
    
    # Étape 1: Chargement de l'image
    try:
        original_img = Image.open(image_path)
        print(f"✓ Image chargée: {original_img.size} (mode: {original_img.mode})")
    except Exception as e:
        print(f"✗ Erreur de chargement de l'image: {e}")
        return
    
    # Étape 2: Redimensionnement
    try:
        ratio = min(target_size / max(original_img.size), 1.0 if max(original_img.size) > target_size else 10.0)
        new_size = (int(original_img.size[0] * ratio), int(original_img.size[1] * ratio))
        resized_img = original_img.resize(new_size, Image.LANCZOS)
        print(f"✓ Image redimensionnée: {resized_img.size} (ratio: {ratio:.3f})")
    except Exception as e:
        print(f"✗ Erreur de redimensionnement: {e}")
        return
    
    # Étape 3: Calcul HOG
    try:
        _, hog_visualization = compute_hog(resized_img)
        hog_stats = {
            'shape': hog_visualization.shape,
            'min': hog_visualization.min(),
            'max': hog_visualization.max(),
            'mean': hog_visualization.mean(),
            'std': hog_visualization.std()
        }
        print(f"✓ HOG calculé: {hog_stats['shape']} (min: {hog_stats['min']}, max: {hog_stats['max']}, "
              f"mean: {hog_stats['mean']:.2f}, std: {hog_stats['std']:.2f})")
        
        # Vérification de qualité du HOG
        if hog_stats['std'] < 10:
            print(f"⚠ Faible variance dans le HOG ({hog_stats['std']:.2f}). "
                  f"Les caractéristiques morphologiques pourraient être insuffisantes.")
    except Exception as e:
        print(f"✗ Erreur de calcul HOG: {e}")
        return
    
    # Étape 4: Dynamic Transform complet
    try:
        transformed_tensor, metadata = dynamic_transform(original_img, add_metadata=True)
        tensor_stats = {
            'shape': transformed_tensor.shape,
            'rgb_min': transformed_tensor[:3].min().item(),
            'rgb_max': transformed_tensor[:3].max().item(),
            'hog_min': transformed_tensor[3].min().item(),
            'hog_max': transformed_tensor[3].max().item(),
            'hog_mean': transformed_tensor[3].mean().item(),
            'hog_std': transformed_tensor[3].std().item()
        }
        print(f"✓ Transformation complète: {tensor_stats['shape']}")
        print(f"  - Canaux RGB: min={tensor_stats['rgb_min']:.3f}, max={tensor_stats['rgb_max']:.3f}")
        print(f"  - Canal HOG: min={tensor_stats['hog_min']:.3f}, max={tensor_stats['hog_max']:.3f}, "
              f"mean={tensor_stats['hog_mean']:.3f}, std={tensor_stats['hog_std']:.3f}")
        
        # Vérifications critiques
        if tensor_stats['shape'][0] != 4:
            print(f"⚠ Nombre incorrect de canaux: {tensor_stats['shape'][0]} (attendu: 4)")
        
        if tensor_stats['hog_std'] < 0.05:
            print(f"⚠ Canal HOG potentiellement dégradé (std={tensor_stats['hog_std']:.3f}). "
                  f"L'identification taxonomique pourrait être affectée.")
    except Exception as e:
        print(f"✗ Erreur de transformation: {e}")
        return
    
    # Visualisation pour débogage
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(original_img)
    plt.title("Image originale")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(hog_visualization, cmap='gray')
    plt.title("Visualisation HOG")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(transformed_tensor[3].numpy(), cmap='jet')
    plt.title("Canal HOG (tenseur)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Validation du pipeline terminée")
    return transformed_tensor, metadata

#=======================================================================================
# TRAITEMENT DES DONNÉES ET DATASETS
#=======================================================================================

class CollemboleDataProcessor:
    """Traite les images et annotations de collemboles pour l'entraînement"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = []
        self.labels_files = []
        self.crops = []
        self.crop_labels = []
        self.expert_labels_list = []
        self.original_sizes = []
        self.project_names = []
        self.agreement_patterns = []
        
        # Nouveaux attributs pour l'augmentation avancée
        self.original_image_paths = []  # Chemin vers l'image originale pour chaque crop
        self.original_bbox_coords = []  # Coordonnées originales de chaque bounding box (x1, y1, x2, y2)

    def find_files(self):
        """Trouve tous les fichiers image et leurs fichiers texte correspondants"""
        for file in os.listdir(self.data_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(self.data_dir, file)
                txt_path = os.path.join(self.data_dir, os.path.splitext(file)[0] + '.txt')

                if os.path.exists(txt_path):
                    self.image_files.append(img_path)
                    self.labels_files.append(txt_path)

        print(f"Found {len(self.image_files)} images with corresponding label files")

    def parse_label_file(self, label_file):
        """Parse les fichiers de labels au format YOLO+ et retourne les bounding boxes"""
        boxes = []
        labels = []
        expert_labels_list = []
        project_names = []
        
        # Statistiques de parsing pour diagnostics
        parsing_stats = {
            'total_lines': 0,
            'valid_lines': 0,
            'invalid_format': 0,
            'invalid_coord': 0,
            'invalid_expert_labels': 0
        }

        def is_valid_float(value):
            """Vérifie si une chaîne peut être convertie en nombre flottant valide"""
            if not isinstance(value, str):
                return False
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False

        with open(label_file, 'r') as f:
            for line_idx, line in enumerate(f, 1):
                parsing_stats['total_lines'] += 1
                parts = line.strip().split()

                # Vérifier qu'il y a au moins 5 parties (labels + 4 coordonnées)
                if len(parts) < 5:
                    parsing_stats['invalid_format'] += 1
                    continue

                # Vérification explicite que les 4 dernières valeurs sont numériques
                coord_parts = parts[-4:]
                if not all(is_valid_float(coord) for coord in coord_parts):
                    parsing_stats['invalid_coord'] += 1
                    continue

                # Le premier élément contient les labels des experts
                expert_labels = parts[0].split('_')
                if len(expert_labels) != 4:
                    parsing_stats['invalid_expert_labels'] += 1
                    continue

                # Convertir les labels des experts en entiers
                try:
                    expert_labels = [int(label) for label in expert_labels]
                    agreement_pattern = get_agreement_type(expert_labels)

                    # Filtrer les cas sans consensus clair
                    if agreement_pattern in ["2/2", "1/1/1/1"]:
                        continue
                    
                    # Ignorer les annotations étiquetées FOND avec désaccord
                    if expert_labels.count(8) >= 3 and expert_labels.count(8) < 4:
                        # Pattern 3/1 avec FOND majoritaire
                        continue
                    elif expert_labels.count(8) == 2:
                        # Pattern 2/1/1 ou 2/2 avec FOND présent
                        continue
                    elif expert_labels.count(8) == 1 and len(set(expert_labels)) == 4:
                        # Pattern 1/1/1/1 avec un vote FOND
                        continue

                    expert_labels_list.append(expert_labels)
                except ValueError:
                    parsing_stats['invalid_expert_labels'] += 1
                    continue

                # Déterminer le label majoritaire
                majority_label = resolve_label_disagreement(expert_labels)

                # Les 4 derniers éléments sont toujours les coordonnées
                # Nous avons déjà vérifié qu'ils sont numériques, donc la conversion est sûre
                try:
                    x_center = float(parts[-4])
                    y_center = float(parts[-3])
                    width = float(parts[-2])
                    height = float(parts[-1])

                    # Vérification complète de la validité des coordonnées:
                    # 1. Dans les plages valides (0-1)
                    # 2. Vérifier que le rectangle ne déborde pas de l'image
                    if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                        0 < width <= 1 and 0 < height <= 1 and
                        x_center - width/2 >= 0 and x_center + width/2 <= 1 and
                        y_center - height/2 >= 0 and y_center + height/2 <= 1):
                        
                        boxes.append([x_center, y_center, width, height])
                        labels.append(majority_label)

                        # Extraction robuste du nom de projet avec gestion des cas limites
                        if len(parts) == 5:  # Format minimal: labels + 4 coords (sans nom de projet)
                            project_name = "Sans_Projet"
                        elif len(parts) > 5:
                            # Format standard avec potentiellement des espaces dans le nom du projet
                            project_name = ' '.join(parts[1:-4])
                        else:
                            # Format incorrect, assigner une valeur par défaut
                            project_name = "Projet_Inconnu"

                        # Nettoyage et normalisation du nom de projet
                        project_name = project_name.strip()
                        if not project_name:  # Si chaîne vide après nettoyage
                            project_name = "Projet_Inconnu"
                        elif '/' in project_name:
                            project_name = project_name.split('/')[0]  # Prendre uniquement le projet principal

                        project_names.append(project_name)
                        parsing_stats['valid_lines'] += 1
                    else:
                        parsing_stats['invalid_coord'] += 1
                except ValueError:
                    # Ce cas ne devrait plus se produire avec la validation préalable,
                    # mais conservé par précaution
                    parsing_stats['invalid_coord'] += 1
                    continue

        # Afficher les statistiques de parsing en mode debug
        print(f"Parsing de {os.path.basename(label_file)}: "
             f"{parsing_stats['valid_lines']}/{parsing_stats['total_lines']} lignes valides "
             f"({parsing_stats['valid_lines']/max(1, parsing_stats['total_lines'])*100:.1f}%)")

        return boxes, labels, expert_labels_list, project_names


    def analyze_project_distribution(self):
        """
        Analyse la distribution des projets dans le dataset actuel.
        
        Returns:
            dict: Dictionnaire {projet: pourcentage} indiquant la proportion de chaque projet
        """
        # Compter les occurrences de chaque projet
        project_counts = {}
        for project in self.project_names:
            if project not in project_counts:
                project_counts[project] = 0
            project_counts[project] += 1
        
        # Calculer les pourcentages
        total_samples = len(self.project_names)
        project_distribution = {
            project: count / total_samples * 100 
            for project, count in project_counts.items()
        }
        
        # Afficher la distribution pour information
        print("\nDistribution des projets dans le dataset:")
        for project, percentage in sorted(project_distribution.items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"  {project}: {percentage:.2f}% ({project_counts[project]} échantillons)")
        
        return project_distribution, project_counts


    def generate_background_samples(self, image_file, boxes, num_samples=1):
        """Génère des échantillons de fond qui ne chevauchent pas les boîtes existantes"""
        try:
            # Ouvrir l'image
            img_pil = Image.open(image_file)
            img_width, img_height = img_pil.size
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Pas de boîtes = pas d'échantillons
            if not boxes:
                return [], []
            
            # Convertir les coordonnées relatives en absolues avec une marge de sécurité
            abs_boxes = []
            box_dimensions = []  # Stocker les dimensions exactes des collemboles

            # Valeur négative = inclusion de parties de collemboles
            # Valeur positive = exclusion avec marge
            for box in boxes:
                x_center, y_center, width, height = box
                # Calcul des coordonnées en pixels
                box_width = int(width * img_width)
                box_height = int(height * img_height)
                
                # Stocker les dimensions exactes pour réutilisation
                box_dimensions.append((box_width, box_height))

                # Marge aléatoire : entre -10 et +20 pixels
                # Probabilité de 20% d'avoir une marge négative (inclure partie du collembole)
                if random.random() < 0.2:
                    safety_margin = random.randint(-10, 0)  # Marge négative (inclusion)
                else:
                    safety_margin = random.randint(0, 20)    # Marge positive (exclusion)
                
                # Calcul des coordonnées de la boîte
                raw_x1 = int((x_center - width/2) * img_width) - safety_margin
                raw_y1 = int((y_center - height/2) * img_height) - safety_margin
                raw_x2 = int((x_center + width/2) * img_width) + safety_margin
                raw_y2 = int((y_center + height/2) * img_height) + safety_margin
                
                # S'assurer que les coordonnées restent dans les limites de l'image
                x1 = max(0, raw_x1)
                y1 = max(0, raw_y1)
                x2 = min(img_width, raw_x2)
                y2 = min(img_height, raw_y2)
                
                # Validation critique : s'assurer que la largeur et la hauteur sont positives
                if x2 <= x1 or y2 <= y1:
                    # Ignorer cette boîte si les dimensions seraient invalides
                    continue
                    
                abs_boxes.append((x1, y1, x2, y2, box_width, box_height))

            # Créer un masque pour les zones déjà occupées
            occupied_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            for x1, y1, x2, y2, _, _ in abs_boxes:
                occupied_mask[y1:y2, x1:x2] = 255
                
            # Dilater le masque pour créer une zone tampon
            kernel = np.ones((safety_margin*2, safety_margin*2), np.uint8)
            occupied_mask = cv2.dilate(occupied_mask, kernel, iterations=1)

            background_crops = []
            background_sizes = []
            samples_created = 0
            max_attempts = 50
            total_attempts = 0

            while samples_created < num_samples and total_attempts < 100:
                total_attempts += 1
                
                # MODIFICATION CLÉ: Choisir une dimension de collembole au hasard
                chosen_idx = random.randint(0, len(box_dimensions) - 1)
                crop_w, crop_h = box_dimensions[chosen_idx]

                # Vérifier que l'image est assez grande
                if crop_w >= img_width or crop_h >= img_height:
                    continue

                # Générer une position aléatoire
                x = random.randint(0, img_width - crop_w)
                y = random.randint(0, img_height - crop_h)
                
                # Vérifier si la région est libre en utilisant le masque
                region = occupied_mask[y:y+crop_h, x:x+crop_w]
                if np.any(region > 0):  # Si un pixel est occupé, rejeter
                    continue
                    
                try:
                    # Extraire la zone de fond
                    crop_img_cv = img_cv[y:y+crop_h, x:x+crop_w]

                    # Vérifications de base
                    if crop_img_cv.shape[0] == 0 or crop_img_cv.shape[1] == 0:
                        continue

                    # Convertir de OpenCV à PIL
                    crop_img = Image.fromarray(cv2.cvtColor(crop_img_cv, cv2.COLOR_BGR2RGB))
                    background_crops.append(crop_img)
                    background_sizes.append((crop_w, crop_h))
                    samples_created += 1

                except Exception as e:
                    print(f"Erreur lors de l'extraction: {str(e)}")
                    continue

            return background_crops, background_sizes

        except Exception as e:
            print(f"Exception générale dans generate_background_samples: {str(e)}")
            return [], []


    def visualize_and_save_background_samples(self, output_dir="background_samples"):
        """
        Visualise et sauvegarde les derniers échantillons de fond générés.
        
        Args:
            output_dir (str): Répertoire où sauvegarder les images de fond
        """
        import os
        import matplotlib.pyplot as plt
        from datetime import datetime
        import numpy as np
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Trouver tous les indices des échantillons de fond (classe 8)
        background_indices = [i for i, label in enumerate(self.crop_labels) if label == 8]
        
        # S'il n'y a pas d'échantillons de fond, informer et quitter
        if not background_indices:
            print("Aucun échantillon de fond n'a été généré.")
            return
        
        # Prendre les 20 derniers échantillons (ou moins s'il y en a moins de 20)
        last_n = min(20, len(background_indices))
        selected_indices = background_indices[-last_n:]
        
        print(f"Affichage des {last_n} derniers échantillons de fond générés:")
        
        # Créer une figure pour afficher les échantillons
        plt.figure(figsize=(15, last_n * 3))
        
        # Pour chaque échantillon sélectionné
        for i, idx in enumerate(selected_indices):
            # Récupérer l'image et les métadonnées
            img = self.crops[idx]
            size = self.original_sizes[idx] if idx < len(self.original_sizes) else "Inconnu"
            experts = self.expert_labels_list[idx] if idx < len(self.expert_labels_list) else "Inconnu"
            pattern = self.agreement_patterns[idx] if idx < len(self.agreement_patterns) else "Inconnu"
            project = self.project_names[idx] if idx < len(self.project_names) else "Inconnu"
            
            # Créer un subplot pour cette image
            plt.subplot(last_n, 1, i+1)
            plt.imshow(np.array(img))
            
            # Ajouter des informations dans le titre
            plt.title(f"Échantillon de fond #{idx}\n"
                    f"Taille: {size}, Consensus: {pattern}\n"
                    f"Labels experts: {experts}, Projet: {project}",
                    fontsize=10)
            plt.axis('off')
            
            # Sauvegarder l'image individuellement
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f"background_sample_{idx}_{timestamp}.png")
            img.save(save_path)
            print(f"Image sauvegardée: {save_path}")
        
        # Ajuster la mise en page et afficher
        plt.tight_layout()
        
        # Sauvegarder la figure complète
        fig_path = os.path.join(output_dir, f"background_samples_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(fig_path)
        print(f"Vue d'ensemble sauvegardée: {fig_path}")
        
        plt.show()
        
        # Afficher également des statistiques sur les échantillons de fond
        total_bg = len(background_indices)
        bg_percentage = (total_bg / len(self.crop_labels)) * 100
        print(f"\nStatistiques des échantillons de fond:")
        print(f"- Nombre total d'échantillons de fond: {total_bg}")
        print(f"- Pourcentage dans le jeu de données: {bg_percentage:.2f}%")
        print(f"- Tailles des échantillons de fond: min={min([self.original_sizes[i] for i in background_indices])}, "
              f"max={max([self.original_sizes[i] for i in background_indices])}")
        
        # Vérifier que tous les échantillons de fond ont un consensus parfait
        consensus_check = all(self.agreement_patterns[i] == "4/4" for i in background_indices)
        print(f"- Tous les échantillons ont-ils un consensus parfait (4/4)? {'Oui' if consensus_check else 'Non'}")


    def crop_images(self):
        """Extrait les crops des images en utilisant les bounding boxes et pré-calcule les HOG"""
        self.find_files()

        global PROJECT_MAPPING
        all_projects = set()

        # Dictionnaire pour collecter les images par projet - AJOUT DE CETTE PARTIE
        images_by_project = {}
        
        # Premier passage: traiter les crops principaux et collecter les images par projet
        for img_path, label_path in tqdm(zip(self.image_files, self.labels_files),
                                        total=len(self.image_files)):
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size

                boxes, labels, expert_labels_list, project_names = self.parse_label_file(label_path)

                # Collecter tous les projets pour le mapping
                all_projects.update(project_names)
                
                # AJOUT: Collecter les images par projet pour la génération de fond équilibrée
                for project_name in set(project_names):
                    if project_name not in images_by_project:
                        images_by_project[project_name] = []
                    images_by_project[project_name].append((img_path, label_path, boxes))

                for box, label, expert_labels, project_name in zip(boxes, labels,
                                                                expert_labels_list,
                                                                project_names):
                    # Calculer le pattern d'accord
                    agreement_pattern = get_agreement_type(expert_labels)

                    # Exclusion pour les patterns indésirables
                    if agreement_pattern in ["2/2", "1/1/1/1"]:
                        continue

                    # Convert relative coordinates to absolute
                    x_center, y_center, width, height = box
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height

                    # Vérifier que width et height sont positifs
                    if width <= 0 or height <= 0:
                        continue

                    # Calculate box coordinates
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)

                    # S'assurer que x2 > x1 et y2 > y1
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Crop the image sans redimensionnement
                    crop = img.crop((x1, y1, x2, y2))

                    # Stocker les informations
                    self.crops.append(crop)
                    self.crop_labels.append(label)
                    self.expert_labels_list.append(expert_labels)
                    self.original_sizes.append((x2 - x1, y2 - y1))
                    self.project_names.append(project_name)
                    self.agreement_patterns.append(agreement_pattern)
                    
                    # Stocker les informations pour l'augmentation optimisée
                    self.original_image_paths.append(img_path)
                    self.original_bbox_coords.append((x1, y1, x2, y2))

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Créer le mapping des projets
        sorted_projects = sorted(list(all_projects))
        PROJECT_MAPPING = {project: idx for idx, project in enumerate(sorted_projects)}
        PROJECT_MAPPING["BACKGROUND"] = len(PROJECT_MAPPING)

        print(f"Generated {len(self.crops)} crops")

        # Afficher la distribution des classes
        unique_labels, counts = np.unique(self.crop_labels, return_counts=True)
        print("Class distribution:")
        for label, count in zip(unique_labels, counts):
            class_name = CLASS_MAPPING.get(label, f"Class {label}")
            print(f"  {class_name} (ID: {label}): {count} samples")

        # Vérifier si nous avons besoin de générer des fonds
        current_bg_count = self.crop_labels.count(8)
        if current_bg_count < BASE_COUNT:
            # Calculer combien d'échantillons de fond à générer
            bg_to_generate = BASE_COUNT - current_bg_count
            print(f"\nGénération de {bg_to_generate} échantillons de fond supplémentaires...")
            
            # Analyser la distribution actuelle des projets (sans compter les fonds)
            project_counts = {}
            for i, (label, project) in enumerate(zip(self.crop_labels, self.project_names)):
                if label != 8:  # Exclure les fonds existants
                    if project not in project_counts:
                        project_counts[project] = 0
                    project_counts[project] += 1
            
            total_non_bg = sum(project_counts.values())
            
            # Calculer combien d'échantillons à générer par projet
            bg_per_project = {}
            for project, count in project_counts.items():
                project_pct = count / total_non_bg * 100
                bg_per_project[project] = max(1, int(bg_to_generate * (project_pct / 100)))
            
            # Ajuster pour atteindre exactement le nombre cible
            total_assigned = sum(bg_per_project.values())
            if total_assigned < bg_to_generate:
                # Distribuer les échantillons restants aux projets les plus importants
                remaining = bg_to_generate - total_assigned
                sorted_projects = sorted(project_counts.items(), key=lambda x: x[1], reverse=True)
                for i in range(remaining):
                    if i < len(sorted_projects):
                        bg_per_project[sorted_projects[i][0]] += 1
            
            # Afficher le plan de génération
            print("\nPlan de génération d'échantillons de fond par projet:")
            for project, count in sorted(bg_per_project.items(), key=lambda x: x[1], reverse=True):
                print(f"  {project}: {count} échantillons à générer "
                    f"({project_counts.get(project, 0)} échantillons existants)")
            
            # Générer les échantillons de fond par projet
            total_generated = 0
            
            for project, target_count in bg_per_project.items():
                if project not in images_by_project or not images_by_project[project]:
                    print(f"  {project}: Aucune image disponible pour générer des fonds")
                    continue
                    
                print(f"  Génération pour {project}: {target_count} échantillons...")
                
                # Compter les échantillons générés pour ce projet
                generated_for_project = 0
                max_attempts = target_count * 3  # Limiter les tentatives
                attempts = 0
                
                # Sélectionner des images aléatoirement pour ce projet
                project_images = images_by_project[project]
                
                while generated_for_project < target_count and attempts < max_attempts:
                    attempts += 1
                    
                    # Sélectionner une image aléatoire avec ses boîtes
                    img_path, label_path, boxes = random.choice(project_images)
                    
                    if not boxes:
                        continue  # Pas de boîtes, impossible de générer un fond
                    
                    # Utiliser la fonction existante pour générer un échantillon de fond
                    bg_samples, bg_sizes = self.generate_background_samples(
                        img_path, boxes, num_samples=1
                    )
                    
                    # Ajouter l'échantillon s'il est valide
                    if bg_samples and bg_sizes:
                        for bg_sample, bg_size in zip(bg_samples, bg_sizes):
                            self.crops.append(bg_sample)
                            self.crop_labels.append(8)  # Classe 8 = FOND
                            self.expert_labels_list.append([8, 8, 8, 8])
                            self.original_sizes.append(bg_size)
                            self.project_names.append(project)  # Conserver le projet d'origine
                            self.agreement_patterns.append("4/4")
                            
                            # Valeurs par défaut pour les autres métadonnées
                            self.original_image_paths.append(img_path)
                            self.original_bbox_coords.append((0, 0, 0, 0))  # Valeurs fictives
                            
                            generated_for_project += 1
                            total_generated += 1
                            
                            if generated_for_project >= target_count:
                                break
                
                print(f"    {project}: {generated_for_project}/{target_count} échantillons générés "
                    f"({attempts} tentatives)")
            
            print(f"\nTotal: {total_generated}/{bg_to_generate} échantillons de fond générés")

        # Afficher la distribution des patterns d'accord
        unique_patterns, pattern_counts = np.unique(self.agreement_patterns, return_counts=True)
        print("\nAgreement pattern distribution:")
        for pattern, count in zip(unique_patterns, pattern_counts):
            print(f"  {pattern}: {count} samples ({count/len(self.agreement_patterns)*100:.1f}%)")
        
        # Modifier la ligne de retour pour enlever self.crop_hogs:
        return (self.crops, self.crop_labels, self.expert_labels_list,
                self.original_sizes, self.project_names, self.agreement_patterns,
                self.original_image_paths, self.original_bbox_coords)
            
    def visualize_samples(self, n_samples=5):
        """Visualize random sample crops with their labels"""
        if not self.crops:
            print("No crops available. Run crop_images() first.")
            return

        # Sélectionner des échantillons de chaque classe si possible
        samples_by_class = {}
        for i, label in enumerate(self.crop_labels):
            if label not in samples_by_class:
                samples_by_class[label] = []
            if len(samples_by_class[label]) < n_samples:
                samples_by_class[label].append(i)

        # Créer une liste de tous les échantillons à visualiser
        all_samples = []
        for label, samples in samples_by_class.items():
            all_samples.extend(samples[:min(n_samples, len(samples))])

        # Si trop d'échantillons, sélectionner aléatoirement
        if len(all_samples) > n_samples * len(samples_by_class):
            all_samples = random.sample(all_samples, n_samples * len(samples_by_class))

        plt.figure(figsize=(15, 3 * ((len(all_samples) + 4) // 5)))
        for i, idx in enumerate(all_samples):
            plt.subplot(((len(all_samples) + 4) // 5), 5, i+1)
            img = np.array(self.crops[idx])
            plt.imshow(img)

            label = self.crop_labels[idx]
            class_name = CLASS_MAPPING.get(label, f"Class {label}")
            size = self.original_sizes[idx] if idx < len(self.original_sizes) else "Unknown"
            project = self.project_names[idx] if idx < len(self.project_names) else "Unknown"
            agreement = self.agreement_patterns[idx] if idx < len(self.agreement_patterns) else "Unknown"

            plt.title(f"{class_name} (ID: {label})\nSize: {size}\nProject: {project}\nAgreement: {agreement}",
                    fontsize=8)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

def balance_dataset(crops, labels, expert_labels_list=None, original_sizes=None,
                    project_names=None, agreement_patterns=None, original_image_paths=None,
                    original_bbox_coords=None, crop_hogs=None, base_count=100, augment=True):
    """
    Équilibre les données en conservant les classes majoritaires et en augmentant
    uniquement les classes minoritaires jusqu'à un seuil minimum.
    
    Version modifiée pour ne plus utiliser les HOG pré-calculés.
    """
    # Vérifier si les métadonnées existent
    has_expert_labels = expert_labels_list is not None and len(expert_labels_list) == len(labels)
    has_sizes = original_sizes is not None and len(original_sizes) == len(labels)
    has_projects = project_names is not None and len(project_names) == len(labels)
    has_patterns = agreement_patterns is not None and len(agreement_patterns) == len(labels)
    has_original_image_paths = original_image_paths is not None and len(original_image_paths) == len(labels)
    has_original_bbox_coords = original_bbox_coords is not None and len(original_bbox_coords) == len(labels)
    has_hogs = crop_hogs is not None and len(crop_hogs) == len(labels)

    # Convertir les labels en array pour faciliter le traitement
    labels_array = np.array(labels)

    # Récupérer les indices par classe en excluant les patterns indésirables
    class_indices = {}
    for label in np.unique(labels_array):
        if has_patterns:
            # Filtrer les patterns indésirables
            indices = [i for i, (l, p) in enumerate(zip(labels, agreement_patterns))
                      if l == label and p not in ["2/2", "1/1/1/1"]]
        else:
            indices = [i for i, l in enumerate(labels) if l == label]

        class_indices[label] = indices

    # Compter les échantillons valides par classe
    class_counts = {k: len(v) for k, v in class_indices.items()}

    print("Distribution originale des classes:")
    for label, count in class_counts.items():
        class_name = CLASS_MAPPING.get(label, f"Class {label}")
        print(f"  {class_name} (ID: {label}): {count} échantillons")

    # Calculer les cibles pour chaque classe: conserver les majoritaires, augmenter les minoritaires
    target_counts = {label: max(count, base_count) for label, count in class_counts.items()}

    # Traitement spécial pour la classe FOND (8) - S'assurer qu'elle atteint au moins base_count*2
    if 8 in class_counts:
        target_counts[8] = max(target_counts[8], base_count*2)
        print(f"  Traitement spécial pour FOND: objectif fixé à {target_counts[8]} échantillons")

    print(f"Équilibrage des classes avec seuil minimum de {base_count} échantillons:")
    for label, target in target_counts.items():
        original = class_counts[label]
        class_name = CLASS_MAPPING.get(label, f"Class {label}")

        if original < base_count:
            print(f"  {class_name} (ID: {label}): Augmentation de {original} à {target}")
        else:
            print(f"  {class_name} (ID: {label}): Conservation des {original} échantillons")

    # Créer le jeu de données équilibré
    balanced_crops = []
    balanced_labels = []
    balanced_expert_labels = [] if has_expert_labels else None
    balanced_sizes = [] if has_sizes else None
    balanced_projects = [] if has_projects else None
    balanced_patterns = [] if has_patterns else None

    # Fonction pour ajouter un échantillon original (sans augmentation)
    def add_original_sample(idx):
        balanced_crops.append(crops[idx])
        balanced_labels.append(labels[idx])

        if has_expert_labels:
            balanced_expert_labels.append(expert_labels_list[idx])
        if has_sizes:
            balanced_sizes.append(original_sizes[idx])
        if has_projects:
            balanced_projects.append(project_names[idx])
        if has_patterns:
            balanced_patterns.append(agreement_patterns[idx])

    # Fonction pour valider un crop augmenté
    def validate_augmented_crop(crop, original_crop):
        """Vérifie que le crop augmenté respecte les critères morphologiques minimaux"""
        # Si le crop est invalide, retourne False
        if crop is None:
            return False
            
        # Vérifier les dimensions minimales
        min_size = 20  # Taille minimale en pixels
        if crop.width < min_size or crop.height < min_size:
            return False
            
        # Vérifier le ratio d'aspect (éviter les crops trop déformés)
        original_ratio = original_crop.width / max(1, original_crop.height)
        new_ratio = crop.width / max(1, crop.height)
        
        # Le nouveau ratio ne doit pas être trop différent de l'original
        ratio_change = abs(new_ratio - original_ratio) / max(original_ratio, 1e-5)
        if ratio_change > 0.5:  # Plus de 50% de changement dans le ratio
            return False
            
        # Vérifier le contraste minimal (critère morphologique important)
        try:
            crop_array = np.array(crop.convert('L'))
            contrast = crop_array.std()
            if contrast < 10:  # Valeur empirique pour un contraste minimal
                return False
        except Exception:
            # En cas d'erreur dans le calcul du contraste, accepter quand même
            pass
            
        # Toutes les vérifications passées
        return True

    def add_augmented_sample(source_idx, augmented_img):
        """Ajoute un échantillon augmenté après validation morphologique"""
        # Vérifier que l'augmentation a produit une image valide
        if not isinstance(augmented_img, Image.Image):
            print(f"Augmentation invalide pour l'échantillon {source_idx}: ce n'est pas une image PIL")
            add_original_sample(source_idx)  # Fallback
            return False
            
        # Validation morphologique
        if not validate_augmented_crop(augmented_img, crops[source_idx]):
            # print(f"Augmentation rejetée: critères morphologiques non respectés")
            add_original_sample(source_idx)  # Fallback
            return False
        
        # L'image a passé la validation, l'ajouter au dataset
        balanced_crops.append(augmented_img)
        balanced_labels.append(labels[source_idx])

        if has_expert_labels:
            balanced_expert_labels.append(expert_labels_list[source_idx])
        if has_sizes:
            # Utiliser directement les dimensions du nouveau crop
            w, h = augmented_img.width, augmented_img.height
            balanced_sizes.append((w, h))
        if has_projects:
            balanced_projects.append(project_names[source_idx])
        if has_patterns:
            balanced_patterns.append(agreement_patterns[source_idx])
            
        return True  # Augmentation réussie

    # AMELIORATION: Fonction avancée pour augmenter via rotation contrôlée
    def augment_with_controlled_rotation(source_idx, max_attempts=10):
        """
        Applique une rotation contrôlée à l'image originale et extrait un nouveau crop.
        Inclut des validations strictes et plusieurs tentatives si nécessaire.
        """
        # Récupérer les données source
        original_img_path = original_image_paths[source_idx]
        original_bbox = original_bbox_coords[source_idx]
        
        # Vérifier que c'est une bbox valide (pour les fonds notamment)
        if sum(original_bbox) == 0:
            # Cas particulier (exemple: fond) - dupliquer l'original sans augmentation
            add_original_sample(source_idx)
            return True
        
        # Plusieurs tentatives d'augmentation
        for attempt in range(max_attempts):
            try:
                # Charger l'image complète originale
                original_img = Image.open(original_img_path)
                img_width, img_height = original_img.size
                
                # Permettre des rotations complètes pour toutes les classes
                angle = random.randint(0, 360)
                
                # Appliquer la rotation avec méthode optimisée pour la qualité
                if angle != 0:
                    # Utiliser BICUBIC pour une meilleure qualité d'interpolation
                    rotated_img = original_img.rotate(angle, 
                                                     resample=Image.BICUBIC,
                                                     expand=False,
                                                     fillcolor=IMAGENET_MEAN_PIXEL)
                else:
                    rotated_img = original_img
                
                # Calculer la nouvelle bounding box avec contrôle de débordement
                new_bbox = transform_bbox(original_bbox, angle, (img_width, img_height))
                x1, y1, x2, y2 = new_bbox
                
                # Vérifier que la bbox est valide
                if x2 <= x1 or y2 <= y1 or x1 >= img_width or y1 >= img_height:
                    continue  # Bbox invalide, essayer à nouveau
                
                # CORRECTION: S'assurer que les dimensions sont raisonnables
                min_size = 20  # Taille minimale en pixels
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    continue  # Trop petit, essayer à nouveau
                
                # Extraire le nouveau crop avec contrôle des limites
                cropped = rotated_img.crop((
                    max(0, x1),
                    max(0, y1),
                    min(img_width, x2),
                    min(img_height, y2)
                ))
                
                # AMÉLIORATION: Appliquer des transformations avancées
                # Ces transformations préservent les caractéristiques morphologiques
                augmented_img = cropped
                
                # Variation de luminosité/contraste (subtile)
                if random.random() < 0.7:  # 70% de chance d'appliquer
                    # Luminosité: variation maximale de ±15%
                    brightness_factor = random.uniform(0.85, 1.15)
                    enhancer = ImageEnhance.Brightness(augmented_img)
                    augmented_img = enhancer.enhance(brightness_factor)
                    
                    # Contraste: variation maximale de ±10%
                    contrast_factor = random.uniform(0.95, 1.1)
                    enhancer = ImageEnhance.Contrast(augmented_img)
                    augmented_img = enhancer.enhance(contrast_factor)
                
                # Vérification de la validation
                if validate_augmented_crop(augmented_img, crops[source_idx]):
                    return add_augmented_sample(source_idx, augmented_img)
                    
            except Exception as e:
                print(f"Tentative {attempt+1} échouée: {str(e)}")
                continue
        
        # MODIFICATION: Ne pas ajouter d'échantillon si l'augmentation échoue
        print(f"Toutes les tentatives d'augmentation ont échoué pour l'échantillon {source_idx}")
        return False  # Retourner False au lieu d'ajouter une copie

    # Traiter chaque classe
    for label, indices in class_indices.items():
        current_count = len(indices)
        target_count = target_counts[label]

        # Cas 1: Classe majoritaire (ou exactement à la cible) - conserver tous les échantillons
        if current_count >= base_count:
            for idx in indices:
                add_original_sample(idx)

        # Cas 2: Classe minoritaire - ajouter tous les échantillons puis augmenter
        else:
            # D'abord ajouter tous les échantillons existants
            for idx in indices:
                add_original_sample(idx)

            # Puis augmenter jusqu'à la cible si possible
            if augment:
                samples_to_add = target_count - current_count
                print(f"  Augmentation classe {label}: ajout de {samples_to_add} échantillons")

                # Identifier les échantillons avec consensus pour l'augmentation
                consensus_indices = []
                if has_patterns:
                    # Sélectionner STRICTEMENT les échantillons avec consensus 4/4
                    consensus_indices = [idx for idx in indices if agreement_patterns[idx] == "4/4"]
                else:
                    # Si pas d'information sur les patterns d'accord, ne pas faire d'augmentation
                    consensus_indices = []
                

                # Vérifier que nous avons les données nécessaires pour l'augmentation avancée
                if consensus_indices and samples_to_add > 0 and has_original_image_paths and has_original_bbox_coords:
                    # Compteur pour les augmentations réussies
                    successful_augmentations = 0
                    max_attempts = samples_to_add * 10  # Augmenter le nombre de tentatives (était 3)
                    
                    # Continuer jusqu'à atteindre le nombre cible ou épuiser les tentatives
                    attempt = 0
                    while successful_augmentations < samples_to_add and attempt < max_attempts:
                        attempt += 1
                        
                        # Sélectionner une image source - stratégie pondérée par qualité
                        if has_patterns:
                            # Favoriser les échantillons avec accord 4/4
                            weights = [4.0 if agreement_patterns[idx] == "4/4" else 
                                    2.0 if agreement_patterns[idx] == "3/1" else 
                                    1.0 for idx in consensus_indices]
                            source_idx = random.choices(consensus_indices, weights=weights, k=1)[0]
                        else:
                            source_idx = random.choice(consensus_indices)
                        
                        # Appliquer l'augmentation contrôlée
                        # IMPORTANT: Fonction augment_with_controlled_rotation doit être modifiée 
                        # pour ne PAS ajouter d'échantillon original en cas d'échec
                        augmentation_success = augment_with_controlled_rotation(source_idx)
                        if augmentation_success:
                            successful_augmentations += 1
                    
                    # Afficher le résultat final de l'augmentation
                    print(f"    {successful_augmentations} augmentations réussies sur {samples_to_add} tentées")
                    
                    # SUPPRIMÉ: Ne plus compléter avec des duplications
                    # Accepter que la classe reste sous-représentée si les augmentations échouent
                    if successful_augmentations < samples_to_add:
                        print(f"    ATTENTION: Classe {label} reste sous-représentée avec {current_count + successful_augmentations} échantillons au lieu de {target_count}")
                else:
                    # Mode alternatif sans duplication
                    print(f"  Impossible d'augmenter la classe {label} sans duplications")
                    print(f"  La classe restera sous-représentée avec {current_count} échantillons")

    # Mélanger le jeu de données
    indices = list(range(len(balanced_crops)))
    random.shuffle(indices)

    balanced_crops = [balanced_crops[i] for i in indices]
    balanced_labels = [balanced_labels[i] for i in indices]

    if has_expert_labels:
        balanced_expert_labels = [balanced_expert_labels[i] for i in indices]
    if has_sizes:
        balanced_sizes = [balanced_sizes[i] for i in indices]
    if has_projects:
        balanced_projects = [balanced_projects[i] for i in indices]
    if has_patterns:
        balanced_patterns = [balanced_patterns[i] for i in indices]

    # Vérifier la distribution finale
    final_counts = {}
    for label in balanced_labels:
        if label not in final_counts:
            final_counts[label] = 0
        final_counts[label] += 1

    print("Distribution finale des classes:")
    for label, count in sorted(final_counts.items()):
        class_name = CLASS_MAPPING.get(label, f"Class {label}")
        original = class_counts.get(label, 0)
        print(f"  {class_name} (ID: {label}): {count} échantillons (était: {original})")

    return (balanced_crops, balanced_labels, balanced_expert_labels,
            balanced_sizes, balanced_projects, balanced_patterns)

class SizeNormalizer:
    """
    Normalise les tailles des spécimens en tenant compte du ratio largeur/hauteur,
    une caractéristique taxonomique importante
    """
    def __init__(self, reference_sizes=None):
        self.max_width = 0
        self.max_height = 0
        self.max_ratio = 0

        # Si des tailles de référence sont fournies, calculer les valeurs max
        if reference_sizes:
            self.update_stats(reference_sizes)

    def update_stats(self, sizes):
        """Met à jour les stats avec les nouvelles tailles"""
        # Extraire les valeurs maximales de width et height
        if isinstance(sizes[0], tuple) and len(sizes[0]) >= 2:
            width_values = [s[0] for s in sizes]
            height_values = [s[1] for s in sizes]
            ratio_values = [s[0]/(s[1] + 1e-8) for s in sizes]
        else:
            # Format alternatif possible
            width_values = [s.width if hasattr(s, 'width') else s[0] for s in sizes]
            height_values = [s.height if hasattr(s, 'height') else s[1] for s in sizes]
            ratio_values = [w/(h + 1e-8) for w, h in zip(width_values, height_values)]

        current_max_w = max(width_values)
        current_max_h = max(height_values)
        current_max_ratio = max(ratio_values)

        self.max_width = max(self.max_width, current_max_w)
        self.max_height = max(self.max_height, current_max_h)
        self.max_ratio = max(self.max_ratio, current_max_ratio)

    def normalize(self, size):
        # Gestion de différents formats d'entrée
        if isinstance(size, torch.Tensor):
            width = size[0].item() if size.dim() > 0 else size.item()
            height = size[1].item() if size.dim() > 0 and size.size(0) > 1 else width
            ratio = size[2].item() if size.dim() > 0 and size.size(0) > 2 else width / (height + 1e-8)
        else:
            # Déstructuration avec gestion des formats variables
            if len(size) >= 3:
                width, height, ratio = size
            else:
                width, height = size[:2]
                ratio = width / (height + 1e-8)
        
        # Validation des données pour éviter les valeurs extrêmes
        if not (0 < width <= 1e4 and 0 < height <= 1e4 and 0.01 < ratio < 100):
            print(f"Avertissement: Paramètres de taille invalides: {size}, utilisation de valeurs par défaut")
            width = min(max(width, 1), 1e4)
            height = min(max(height, 1), 1e4)
            ratio = min(max(ratio, 0.01), 100)
        
        # Normalisation avec écrêtage pour garantir des valeurs entre 0 et 1
        width_norm = min(1.0, width / self.max_width) if self.max_width > 0 else 0
        height_norm = min(1.0, height / self.max_height) if self.max_height > 0 else 0
        ratio_norm = min(1.0, ratio / self.max_ratio) if self.max_ratio > 0 else 0

        # Logging si les dimensions dépassent les maximums d'entraînement
        if width > self.max_width or height > self.max_height or ratio > self.max_ratio:
            print(f"Avertissement: Dimensions ({width}x{height}, ratio={ratio:.2f}) dépassent les maximums d'entraînement "
                f"({self.max_width}x{self.max_height}, ratio={self.max_ratio:.2f}). Valeurs écrêtées à 1.0.")

        return torch.tensor([width_norm, height_norm, ratio_norm], dtype=torch.float32)

class CollemboleDataset(Dataset):
    """
    Dataset adapté aux collemboles avec gestion des métadonnées morphologiques,
    préservation des ratios d'aspect et utilisation des HOG pré-calculés
    """
    def __init__(self, images, labels, expert_labels=None, sizes=None, projects=None,
                 patterns=None, hogs=None, transform=None, size_processor=None):
        self.images = images
        self.labels = labels
        self.expert_labels = expert_labels
        self.sizes = sizes
        self.projects = projects
        self.patterns = patterns
        self.transform = transform
        self.size_processor = size_processor

        # Vérifier quelles métadonnées sont disponibles
        self.has_expert_labels = expert_labels is not None and len(expert_labels) == len(labels)
        self.has_sizes = sizes is not None and len(sizes) == len(labels)
        self.has_projects = projects is not None and len(projects) == len(labels)
        self.has_patterns = patterns is not None and len(patterns) == len(labels)

        # Calculer les dimensions maximales localement si nécessaire
        if sizes is not None and size_processor is None:
            width_values = [s[0] if isinstance(s, (tuple, list)) else s for s in sizes]
            height_values = [s[1] if isinstance(s, (tuple, list)) and len(s) > 1 else s for s in sizes]
            self.max_width = max(width_values) if width_values else 3000.0
            self.max_height = max(height_values) if height_values else 3000.0
            self.max_ratio = 5.0  # valeur par défaut pour le ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Récupérer les métadonnées
        expert_label = self.expert_labels[idx] if self.has_expert_labels else None
        size = self.sizes[idx] if self.has_sizes else None
        project = PROJECT_MAPPING.get(self.projects[idx], 0) if self.has_projects else None
        pattern = self.patterns[idx] if self.has_patterns else None
        
        # Appliquer les transformations
        if self.transform:
            transform_result = self.transform(image)
            
            if isinstance(transform_result, tuple) and len(transform_result) == 2:
                # Mode recommandé: HOG déjà calculé par dynamic_transform
                enhanced, transform_info = transform_result
            else:
                enhanced = transform_result
                transform_info = {}
        else:
            # Sans transformation explicite, calculer HOG
            enhanced = dynamic_transform(image)
        
        # Créer un masque basé sur la taille de l'image transformée
        if isinstance(enhanced, torch.Tensor):
            # Utiliser directement les dimensions du tenseur enhanced
            padding_mask = create_padding_mask((enhanced.shape[2], enhanced.shape[1]))
        else:
            padding_mask = None

        # Transformer la taille avec le processeur de taille
        if self.has_sizes and self.size_processor:
            # Gérer différents types de données de taille
            if isinstance(size, torch.Tensor):
                if size.dim() > 0:
                    width = size[0].item()
                    height = size[1].item() if size.size(0) > 1 else width
                else:
                    width = height = size.item()
            else:
                # Supposer un tuple/liste
                width, height = size

            ratio = width / (height + 1e-8)  # Éviter division par zéro
            size_features = self.size_processor.normalize((width, height, ratio))
        elif self.has_sizes:
            # Normalisation basée sur les données via le size_processor
            if isinstance(size, torch.Tensor):
                if size.dim() > 0:
                    width = size[0].item()
                    height = size[1].item() if size.size(0) > 1 else width
                else:
                    width = height = size.item()
            else:
                width, height = size

            ratio = width / (height + 1e-8)  # Éviter division par zéro

            # Utiliser le normaliseur de taille au lieu d'une valeur arbitraire
            if self.size_processor:
                size_features = self.size_processor.normalize((width, height, ratio))
            else:
                # Fallback si le processeur n'est pas disponible
                size_features = torch.tensor([width / self.max_width, height / self.max_height, ratio / 5.0],
                                            dtype=torch.float32) if hasattr(self, 'max_width') else \
                                torch.tensor([width / 3000.0, height / 3000.0, ratio], dtype=torch.float32)
        else:
            size_features = None

        # Créer un dictionnaire avec toutes les métadonnées
        metadata = {}
        if self.has_expert_labels:
            metadata['expert_labels'] = torch.tensor(expert_label, dtype=torch.long)
        if self.has_sizes:
            metadata['size'] = size_features
        if self.has_projects:
            metadata['project'] = torch.tensor(project, dtype=torch.long)
        if self.has_patterns:
            metadata['pattern'] = pattern
        if padding_mask is not None:
            metadata['padding_mask'] = padding_mask

        # Retourner les données et les métadonnées
        return enhanced, label, metadata
    
class TestDataset(Dataset):
    """Dataset pour les images de test, sans annotations"""
    def __init__(self, data_dir, transform=None, size_normalizer=None):
        self.data_dir = data_dir
        self.transform = transform
        self.size_normalizer = size_normalizer
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                           if f.endswith('.jpg') or f.endswith('.png')]
        self.image_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image_id = self.image_ids[idx]

        # Charger l'image sans redimensionnement - préservation cruciale des caractéristiques
        image = Image.open(image_path).convert('RGB')
        original_size = (image.width, image.height)
  
        return image, image_id, original_size


def collate_fn(batch):
    """
    Fonction de collage optimisée pour gérer des images de tailles variables
    avec création efficace de masques de padding et extraction des ROI.
    Gère explicitement les cas avec 3 et 4 canaux (RGB vs RGB+HOG).
    """
    images, labels, metadata_list = zip(*batch)

    if isinstance(images[0], Image.Image):
        # Trouver les dimensions maximales 
        max_raw_w = max(img.width for img in images)
        max_raw_h = max(img.height for img in images)
        max_w = min(max_raw_w, 1024)
        max_h = min(max_raw_h, 1024)

        padded_images = []
        padding_masks = []
        roi_boxes = []  # Liste pour stocker les boîtes ROI

        for img in images:
            # Calculer le ratio pour préserver les proportions
            ratio = min(max_w / img.width, max_h / img.height)

            # Redimensionner si nécessaire
            if ratio < 1.0:
                new_w = int(img.width * ratio)
                new_h = int(img.height * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            # Créer l'image rembourrée et la centrer
            padded = Image.new(img.mode, (INPUT_SIZE, INPUT_SIZE), IMAGENET_MEAN_PIXEL)  # Taille fixe
            x_offset = (INPUT_SIZE - img.width) // 2
            y_offset = (INPUT_SIZE - img.height) // 2
            padded.paste(img, (x_offset, y_offset))

            # CORRECTION: Appliquer dynamic_transform directement
            # Cela gère correctement la conversion et normalisation avec HOG
            transformed = dynamic_transform(padded)
            padded_images.append(transformed)

            # Créer un masque indiquant la zone d'image réelle (sur CPU)
            mask = torch.zeros((1, max_h, max_w))
            mask[:, y_offset:y_offset+img.height, x_offset:x_offset+img.width] = 1
            padding_masks.append(mask)
            
            # Extraire la boîte englobante (ROI)
            roi_box = torch.tensor([[x_offset, y_offset, 
                                    x_offset+img.width, y_offset+img.height]])
            roi_boxes.append(roi_box)

        # CORRECTION: Vérifier les dimensions avant empilage
        # Déterminer le nombre de canaux (3 ou 4) en fonction du premier élément
        num_channels = padded_images[0].size(0)
        
        # S'assurer que tous les tenseurs ont le même nombre de canaux
        for i, tensor in enumerate(padded_images):
            if tensor.size(0) != num_channels:
                print(f"Attention: L'image {i} a {tensor.size(0)} canaux au lieu de {num_channels}")
                # Adapter le tenseur au nombre de canaux attendu
                if tensor.size(0) < num_channels:
                    # Ajouter des canaux manquants (remplis de zéros)
                    padding = torch.zeros((num_channels - tensor.size(0), *tensor.shape[1:]))
                    padded_images[i] = torch.cat([tensor, padding], dim=0)
                else:
                    # Tronquer les canaux excédentaires
                    padded_images[i] = tensor[:num_channels]
        
        # Empiler les images et les masques (sur CPU)
        stacked_images = torch.stack(padded_images)
        stacked_masks = torch.stack(padding_masks)
        
    else:
        # Les images sont déjà des tenseurs
        # CORRECTION: Gestion explicite des dimensions pour les tenseurs existants
        
        # Déterminer le nombre de canaux du premier tenseur
        num_channels = images[0].size(0) if images[0].dim() == 3 else images[0].size(1)
        
        # Vérifier et standardiser tous les tenseurs
        corrected_images = []
        for img in images:
            if img.dim() == 3:  # Format (C, H, W)
                if img.size(0) != num_channels:
                    # Adapter au nombre de canaux standard
                    if img.size(0) < num_channels:
                        padding = torch.zeros((num_channels - img.size(0), *img.shape[1:]))
                        img = torch.cat([img, padding], dim=0)
                    else:
                        img = img[:num_channels]
                corrected_images.append(img)
            elif img.dim() > 3:  # Format incorrect
                print(f"Correction de tenseur avec dimensions: {img.shape}")
                # Extraire les 3 ou 4 premiers canaux selon le standard
                img = img.view(-1, *img.shape[-2:])[:num_channels]
                corrected_images.append(img)
            else:
                # Cas inattendu
                print(f"Format de tenseur non géré: {img.shape}")
                # Créer un tenseur de substitution
                img = torch.zeros((num_channels, 224, 224))
                corrected_images.append(img)
        
        stacked_images = torch.stack(corrected_images)
        stacked_masks = None

    # Convertir et regrouper
    stacked_labels = torch.tensor(labels, dtype=torch.long)
    stacked_metadata = {}
    
    if metadata_list and isinstance(metadata_list[0], dict):
        # Trouver les clés communes à toutes les métadonnées
        all_keys = set(metadata_list[0].keys())
        try:
            common_keys = set.intersection(*[set(m.keys()) for m in metadata_list])
        except Exception:
            common_keys = all_keys  # Fallback si l'intersection échoue
        
        for key in common_keys:
            if key == 'padding_mask' and stacked_masks is not None:
                stacked_metadata[key] = stacked_masks
            elif key == 'roi_box' and all('roi_box' in m for m in metadata_list):
                # Gérer proprement les ROI boxes
                roi_tensors = []
                for m in metadata_list:
                    roi = m[key]
                    if roi.dim() > 2:
                        roi = roi.view(-1, 4)
                    roi_tensors.append(roi)
                stacked_metadata[key] = torch.stack(roi_tensors)
            elif all(isinstance(m[key], torch.Tensor) for m in metadata_list):
                try:
                    stacked_metadata[key] = torch.stack([m[key] for m in metadata_list])
                except RuntimeError as e:
                    print(f"Erreur lors de l'empilement pour '{key}': {e}")
                    stacked_metadata[key] = [m[key] for m in metadata_list]
            else:
                stacked_metadata[key] = [m[key] for m in metadata_list]

    return stacked_images, stacked_labels, stacked_metadata

#=======================================================================================
# FONCTIONS D'ENTRAÎNEMENT ET D'ÉVALUATION
#=======================================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=N_EPOCHS, device='cuda', use_metadata=True):
    
    model = model.to(device)
    best_val_f1 = 0.0
    best_model_weights = None
    early_stopping = EarlyStopping(patience=5, delta=0.001, monitor='f1')
    
    # Assurer un état propre au démarrage
    clear_gpu_memory()
    
    # Activer l'entraînement en précision mixte
    scaler = torch.cuda.amp.GradScaler()
    
    # Fonction de surveillance de la mémoire GPU - uniquement en développement
    def print_gpu_memory(step_name=""):
        """
        Surveille et affiche l'utilisation détaillée de la mémoire GPU.
        
        Args:
            step_name (str): Nom de l'étape actuelle pour le logging
        """
        if torch.cuda.is_available():
            # Forcer le garbage collector pour des mesures plus précises
            gc.collect()
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
            percentage = allocated / max_mem * 100
            
            print(f"GPU Memory [{step_name}]: {allocated:.2f}/{max_mem:.2f} GB allocated ({percentage:.1f}%), "
                f"{reserved:.2f} GB reserved, {reserved - allocated:.2f} GB cache")
            
            # Vérifier les fragments de mémoire (si disponible)
            try:
                stats = torch.cuda.memory_stats()
                if 'allocated_bytes.all.current' in stats and 'reserved_bytes.all.current' in stats:
                    fragmentation = 1.0 - (stats['active_bytes.all.current'] / stats['reserved_bytes.all.current'])
                    print(f"  Fragmentation mémoire: {fragmentation:.1%}")
            except:
                pass

    def run_batch_safely(inputs, labels, metadata, reduction_count=0):
        """
        Exécute un forward/backward en divisant le batch si nécessaire,
        avec gestion optimisée de la mémoire GPU pour éviter les fuites.
        Accumule les gradients sans effectuer de mise à jour des poids.
        """
        if reduction_count > 3:  # Prévenir boucle infinie
            print("ERREUR: Impossible de traiter même un seul élément. Vérifiez votre GPU.")
            raise RuntimeError("Mémoire GPU insuffisante même pour un seul élément")
                
        # Diagnostic avant le traitement du batch
        if reduction_count == 0 and torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"Mémoire GPU avant traitement du batch: {allocated_before:.3f} GB")
                
        try:
            # Préparation des métadonnées sur CPU
            sizes = metadata.get('size', None)
            padding_mask = metadata.get('padding_mask', None)
            patterns = metadata.get('pattern', None)
            
            # CORRECTION: Vérifier et corriger les dimensions d'entrée
            if inputs.dim() > 4:
                print(f"CORRECTION: Redimensionnement de inputs de {inputs.shape} à ", end="")
                b = inputs.size(0)
                inputs = inputs.view(b, 3, inputs.size(-2), inputs.size(-1))
                print(f"{inputs.shape}")
            
            # Transfert GPU uniquement si nécessaire (juste avant calcul)
            with torch.no_grad():
                # Transfert avec non_blocking=True pour paralléliser CPU/GPU
                inputs_gpu = inputs.to(device, non_blocking=True)
                labels_gpu = labels.to(device, non_blocking=True)
                sizes_gpu = sizes.to(device, non_blocking=True) if sizes is not None else None
                padding_mask_gpu = padding_mask.to(device, non_blocking=True) if padding_mask is not None else None
                
                # Calcul des poids pour la focal loss (sur CPU)
                weights = None
                weights_gpu = None
                if patterns is not None:
                    weights = batch_confidence_weight(patterns)
                    weights_gpu = weights.to(device, non_blocking=True)
            
            # Forward pass avec précision mixte
            with torch.cuda.amp.autocast():
                if use_metadata:
                    outputs, uncertainty = model(inputs_gpu, sizes_gpu, padding_mask_gpu)
                else:
                    outputs, uncertainty = model(inputs_gpu)
                
                # Calcul de la perte avec gestion des poids
                if weights_gpu is not None:
                    weights_gpu = torch.clamp(weights_gpu, 0.1, 10.0)  # Éviter les valeurs extrêmes
                    loss = (weights_gpu * criterion(outputs, labels_gpu)).mean()
                else:
                    loss = criterion(outputs, labels_gpu)
            
            # MODIFICATION: Seulement calculer le gradient, sans mise à jour des poids
            # Cela permet d'accumuler les gradients sur les différents sous-batches
            scaler.scale(loss).backward()
            
            # Obtenir la valeur de loss avant de libérer la mémoire
            batch_size = inputs_gpu.size(0)
            loss_value = loss.detach().clone().cpu().item() * batch_size
            
            # Construire le dictionnaire de résultats
            batch_stats = {
                'loss': loss_value,
                'outputs': outputs.detach().clone().cpu(),
                'labels': labels_gpu.clone().cpu(),
                'uncertainty': uncertainty.view(-1).detach().clone().cpu(),
                'was_divided': False
            }
            
            # Libérer les ressources
            del outputs, uncertainty, loss
            del inputs_gpu, labels_gpu
            
            return batch_stats
                
        except RuntimeError as e:
            # Libérer toute la mémoire GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Si erreur OOM, diviser le batch en deux
            if "CUDA out of memory" in str(e) and inputs.size(0) > 1:
                print(f"Divisant le batch de taille {inputs.size(0)} en deux...")
                mid = inputs.size(0) // 2
                
                # Traiter la première moitié - accumule les gradients
                first_half_stats = run_batch_safely(
                    inputs[:mid].clone(),
                    labels[:mid].clone(), 
                    {k: (v[:mid].clone() if isinstance(v, torch.Tensor) else v) 
                    for k, v in metadata.items()},
                    reduction_count + 1
                )
                
                # Traiter la seconde moitié - continue d'accumuler les gradients
                second_half_stats = run_batch_safely(
                    inputs[mid:].clone(),
                    labels[mid:].clone(), 
                    {k: (v[mid:].clone() if isinstance(v, torch.Tensor) else v) 
                    for k, v in metadata.items()},
                    reduction_count + 1
                )
                
                # Combiner les statistiques
                combined_stats = {
                    'loss': first_half_stats['loss'] + second_half_stats['loss'],
                    'outputs': torch.cat([first_half_stats['outputs'], second_half_stats['outputs']]),
                    'labels': torch.cat([first_half_stats['labels'], second_half_stats['labels']]),
                    'uncertainty': torch.cat([first_half_stats['uncertainty'], second_half_stats['uncertainty']]),
                    'was_divided': True
                }
                
                return combined_stats
            else:
                # Si l'erreur n'est pas liée à la mémoire, la propager
                print(f"Erreur non liée à la mémoire: {str(e)}")
                raise
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'train_uncertainty': [],
        'val_uncertainty': []
    }

    for epoch in range(num_epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        train_uncertainties = []
        
        # Surveillance mémoire au début de chaque epoch
        if epoch == 0:
            print_gpu_memory(f"début epoch {epoch+1}")

        for batch_idx, (inputs, labels, metadata) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):

            # DÉBOGAGE: Inspectez les tenseurs au début uniquement pour les premiers lots
            if epoch == 0 and batch_idx < 5:  # Seulement pour les 5 premiers lots du premier epoch
                debug_tensor_shapes(batch_idx, inputs, metadata)
                
            # Mise à zéro des gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Traiter le batch avec sécurité mémoire - accumule seulement les gradients
            batch_stats = run_batch_safely(inputs, labels, metadata)
            
            # CORRECTION: Mise à jour UNIQUE des poids après le traitement de tous les sous-batches
            # Cette mise à jour est maintenant la seule et utilise les gradients accumulés
            scaler.step(optimizer)
            scaler.update()
            
            # Accumuler les statistiques
            train_loss += batch_stats['loss']
            
            with torch.no_grad():
                _, preds = torch.max(batch_stats['outputs'], 1)
                train_preds.extend(preds.numpy())
                train_targets.extend(batch_stats['labels'].numpy())
                train_uncertainties.extend(batch_stats['uncertainty'].numpy())
                
            # Libérer la mémoire du batch
            del batch_stats, inputs, labels
            clear_gpu_memory()
            
            # Surveillance occasionnelle de la mémoire
            if epoch == 0 and batch_idx % 10 == 0:
                print_gpu_memory(f"batch {batch_idx}")

        # Calcul des statistiques d'entraînement
        train_loss = train_loss / len(train_loader.dataset)
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        train_uncertainty = np.mean(train_uncertainties)

        # Phase de validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_uncertainties = []
        
        clear_gpu_memory()  # Nettoyer avant validation

        with torch.no_grad():
            for inputs, labels, metadata in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Préparer les données (sur CPU puis transfert GPU)
                sizes = metadata.get('size', None)
                padding_mask = metadata.get('padding_mask', None)
                
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if sizes is not None:
                    sizes = sizes.to(device, non_blocking=True)
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device, non_blocking=True)

                # Forward pass avec précision mixte
                with torch.cuda.amp.autocast():
                    if use_metadata:
                        outputs, uncertainty = model(inputs, sizes, padding_mask)
                    else:
                        outputs, uncertainty = model(inputs)
                    loss = criterion(outputs, labels)

                # Statistiques
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_uncertainties.extend(uncertainty.view(-1).cpu().numpy())
                
                # Libérer la mémoire GPU
                del inputs, labels, outputs, uncertainty
                if sizes is not None:
                    del sizes
                if padding_mask is not None:
                    del padding_mask
                
                # Nettoyer mémoire GPU entre les batchs de validation
                if batch_idx % 5 == 0:
                    clear_gpu_memory()

        # Calcul des statistiques d'epoch
        val_loss = val_loss / len(val_loader.dataset)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_uncertainty = np.mean(val_uncertainties)

        # Mise à jour du scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Vérification de l'early stopping
        if early_stopping.monitor == 'loss':
            early_stopping(val_loss)
        else:  # Monitor est 'f1'
            early_stopping(val_f1)

        # Sauvegarde du meilleur modèle
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = model.state_dict().copy()

        # Mise à jour de l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_uncertainty'].append(train_uncertainty)
        history['val_uncertainty'].append(val_uncertainty)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train Uncertainty: {train_uncertainty:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Uncertainty: {val_uncertainty:.4f}')

        # Libérer mémoire GPU à la fin de chaque epoch
        clear_gpu_memory()
        
        # Arrêt si early stopping
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Chargement des meilleurs poids
    model.load_state_dict(best_model_weights)

    return model, history

def evaluate_model(model, dataloader, device='cuda', use_metadata=True, agreement_eval=False):
    """
    Évalue le modèle sur un jeu de données et retourne les métriques de performance
    Optimisé pour économiser la mémoire GPU
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_uncertainties = []

    # Pour l'évaluation par type d'accord
    if agreement_eval:
        agreement_results = {
            '4/4': {'correct': 0, 'total': 0},
            '3/1': {'correct': 0, 'total': 0},
            '2/2': {'correct': 0, 'total': 0},
            '2/1/1': {'correct': 0, 'total': 0},
            '1/1/1/1': {'correct': 0, 'total': 0}
        }

    with torch.no_grad():
        for batch_idx, (inputs, labels, metadata) in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Transfert des données vers le GPU juste avant utilisation
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Préparer les métadonnées
            sizes = metadata.get('size', None)
            padding_mask = metadata.get('padding_mask', None)
            patterns = metadata.get('pattern', None) if agreement_eval else None

            if sizes is not None:
                sizes = sizes.to(device, non_blocking=True)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device, non_blocking=True)

            # Forward pass avec métadonnées
            with torch.cuda.amp.autocast():
                if use_metadata:
                    outputs, uncertainty = model(inputs, sizes, padding_mask)
                else:
                    outputs, uncertainty = model(inputs)

                # Extraire les prédictions et les passer immédiatement sur CPU
                _, preds = torch.max(outputs, 1)
                preds_cpu = preds.cpu().numpy()
                labels_cpu = labels.cpu().numpy()
                uncertainty_cpu = uncertainty.view(-1).cpu().numpy()

            # Ajouter les prédictions aux listes (déjà sur CPU)
            all_preds.extend(preds_cpu)
            all_targets.extend(labels_cpu)
            all_uncertainties.extend(uncertainty_cpu)

            # Statistiques par type d'accord
            if agreement_eval and patterns is not None:
                for i, pattern in enumerate(patterns):
                    if pattern in agreement_results:
                        agreement_results[pattern]['total'] += 1
                        agreement_results[pattern]['correct'] += int(preds_cpu[i] == labels_cpu[i])

            # Libérer la mémoire GPU
            del inputs, labels, outputs, uncertainty
            if sizes is not None:
                del sizes
            if padding_mask is not None:
                del padding_mask
            
            # Nettoyer la mémoire GPU périodiquement
            if batch_idx % 5 == 0:
                clear_gpu_memory()

    # Calcul du F1 score global
    f1 = f1_score(all_targets, all_preds, average='macro')
    f1_per_class = f1_score(all_targets, all_preds, average=None)

    # Matrice de confusion
    cm = confusion_matrix(all_targets, all_preds)

    # Ajout du rapport de classification complet pour monitoring détaillé
    target_names = [CLASS_MAPPING.get(i, f"Class {i}") for i in range(len(CLASS_MAPPING))]
    class_report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)

    # Calcul spécifique du recall sur les classes rares (classes avec moins de 5% des échantillons)
    class_counts = np.bincount(all_targets)
    total_samples = len(all_targets)
    rare_classes = [i for i, count in enumerate(class_counts) if count/total_samples < 0.05]
    if rare_classes:
        rare_class_recall = np.mean([class_report[CLASS_MAPPING.get(i, f"Class {i}")]['recall'] for i in rare_classes])
        print(f"Recall moyen sur les classes rares: {rare_class_recall:.4f}")

    # Calcul de l'incertitude moyenne
    mean_uncertainty = np.mean(all_uncertainties)

    # Calcul des scores par type d'accord
    if agreement_eval:
        for pattern in agreement_results:
            if agreement_results[pattern]['total'] > 0:
                agreement_results[pattern]['accuracy'] = agreement_results[pattern]['correct'] / agreement_results[pattern]['total']
                agreement_results[pattern]['count'] = agreement_results[pattern]['total']
            else:
                agreement_results[pattern]['accuracy'] = 0.0
                agreement_results[pattern]['count'] = 0

        return f1, f1_per_class, cm, mean_uncertainty, agreement_results

    return f1, f1_per_class, cm, mean_uncertainty

@contextmanager
def grad_cam_hooks(model, target_layer):
    """Gestionnaire de contexte pour les hooks Grad-CAM avec nettoyage automatique"""
    activations = []
    gradients = []
    
    def fwd_hook(module, input, output):
        activations.append(output.detach())
    
    def bwd_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_backward_hook(bwd_hook)
    
    try:
        yield activations, gradients
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
        torch.cuda.empty_cache()

def generate_heatmap(model, img_tensor, target_class=None, device='cuda'):
    """
    Génère une carte de chaleur Grad-CAM pour l'image d'entrée avec optimisation mémoire.
    Version sécurisée contre les fuites mémoire.
    """
    model.eval()
    
    # Si l'image a déjà 4 canaux, l'utiliser directement
    # Sinon, vérifier s'il faut ajouter un canal supplémentaire (cas rare)
    if img_tensor.size(1) != 4 and img_tensor.dim() > 3:
        print("Attention: L'entrée n'a pas 4 canaux. Adaptation nécessaire pour Grad-CAM.")
        # On pourrait ajouter un 4ème canal factice ici si nécessaire
    
    img_tensor = img_tensor.to(device)

    # Forward pass pour obtenir la prédiction
    with torch.no_grad():
        outputs, _ = model(img_tensor)
        prob = F.softmax(outputs.float(), dim=1)  # Conversion explicite en float32
        pred_class = torch.argmax(prob, dim=1).item()

    # Classe cible (prédite si non spécifiée)
    target_class = pred_class if target_class is None else target_class

    # Trouver la dernière couche de convolution
    target_layer = None

    # Pour notre architecture DynamicFPN, cibler une couche de fusion
    if hasattr(model, 'fusion_convs'):
        target_layer = model.fusion_convs[0]  # Prendre la première couche de fusion

    if target_layer is None:
        return None, pred_class, prob[0, pred_class].item()

    # Utiliser le gestionnaire de contexte pour gérer les hooks en toute sécurité
    with grad_cam_hooks(target_layer) as (activations, gradients):
        # Forward pass avec calcul de gradient
        with torch.enable_grad():
            img_tensor.requires_grad_(True)
            outputs, _ = model(img_tensor)

            # Libérer les sorties non nécessaires
            target_output = outputs[0, target_class]
            prob_val = F.softmax(outputs.float(), dim=1)[0, target_class].item()  # Conversion en float32
            del outputs

            # Backpropagation ciblée sur la classe d'intérêt
            model.zero_grad()
            target_output.backward(retain_graph=False)
        
        # Vérifier que les activations et gradients ont bien été capturés
        if activations is None or gradients is None:
            print("Avertissement: Les activations ou gradients n'ont pas été capturés correctement")
            return None, pred_class, prob_val
            
        # Calcul de la carte de chaleur
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * activations, dim=1).squeeze()

        # Normalisation entre 0 et 1
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-10)

        # Convertir en numpy
        heatmap = heatmap.cpu().numpy()

    # Libérer la mémoire explicitement
    del img_tensor, weights
    clear_gpu_memory()

    return heatmap, pred_class, prob_val

def plot_confusion_matrix(cm, class_mapping):
    """Affiche la matrice de confusion"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_mapping[i] for i in range(len(class_mapping))],
                yticklabels=[class_mapping[i] for i in range(len(class_mapping))])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.show()

def plot_training_history(history):
    """Affiche l'évolution des métriques pendant l'entraînement"""
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('F1 Score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['train_uncertainty'], label='Train Uncertainty')
    plt.plot(history['val_uncertainty'], label='Validation Uncertainty')
    plt.title('Model Uncertainty over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Uncertainty')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.show()

def visualize_heatmap(image, heatmap, pred_class, prob):
    """
    Visualise une image avec sa carte de chaleur superposée pour identifier
    les caractéristiques morphologiques clés utilisées par le modèle.
    """
    # Convertir l'image du format tensor au format numpy si nécessaire
    if isinstance(image, torch.Tensor):
        # Si l'image a 4 canaux (RGB+HOG), extraire uniquement les 3 canaux RGB pour l'affichage
        if image.size(0) == 4:
            image = image[:3]  # Prendre uniquement les canaux RGB
        image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        # Normaliser l'image pour l'affichage
        image = (image - image.min()) / (image.max() - image.min())

    # Redimensionner la heatmap à la taille de l'image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Créer une version colorée de la heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Superposer l'image et la heatmap
    superimposed = 0.6 * image + 0.4 * heatmap_colored
    superimposed = np.clip(superimposed, 0, 1)

    # Créer une figure avec deux sous-plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Afficher l'image originale
    ax1.imshow(image)
    ax1.set_title(f"Image originale\nPrédit: {CLASS_MAPPING[pred_class]} ({prob:.2f})")
    ax1.axis('off')

    # Afficher l'image avec la heatmap
    ax2.imshow(superimposed)
    ax2.set_title(f"Carte de chaleur GradCAM\nCaractéristiques pour {CLASS_MAPPING[pred_class]}")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_{CLASS_MAPPING[pred_class]}.png'))
    plt.show()

def analyze_expert_agreement(data_dir, logger=None):
    """
    Analyse le niveau d'accord entre experts dans les données d'entraînement de collemboles.
    
    Cette fonction examine les annotations de bounding boxes par 4 experts différents et
    quantifie leur niveau d'accord pour chaque spécimen. Elle est essentielle pour évaluer
    la fiabilité des annotations et identifier les classes plus difficiles à reconnaître.
    
    Arguments:
        data_dir (str): Chemin vers le répertoire contenant les fichiers d'annotation (.txt)
        logger (Logger, optional): Logger pour enregistrer les messages et erreurs
    
    Returns:
        tuple: Triplet contenant:
            - agreement_stats (dict): Statistiques globales d'accord (nombres et pourcentages)
            - class_dist_formatted (dict): Distribution des classes pour les annotations avec accord total
            - agreement_by_class_formatted (dict): Statistiques d'accord par classe
    
    Exemple d'utilisation:
        agreement_stats, class_dist, agreement_by_class = analyze_expert_agreement("./data/")
        print(f"Accord total sur {agreement_stats['full_agreement_pct']:.1f}% des annotations")
    """
    # Utiliser le logger fourni ou créer un logger basique
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    # Initialiser les dictionnaires pour stocker les résultats
    agreement_stats = {
        'full_agreement': 0,      # Cas où les 4 experts sont d'accord (4/4)
        'majority_agreement': 0,  # Cas où 3 experts sont d'accord (3/1)
        'split_decision': 0,      # Cas de désaccord plus marqué (2/2, 2/1/1, 1/1/1/1)
        'total_boxes': 0          # Nombre total de bounding boxes analysées
    }

    # Dictionnaire pour suivre la distribution des classes en cas d'accord total
    class_distribution = {}
    
    # Dictionnaire pour analyser l'accord par classe taxonomique
    agreement_by_class = {}

    # Vérifier que le répertoire existe
    if not os.path.isdir(data_dir):
        logger.error(f"Le répertoire {data_dir} n'existe pas")
        return agreement_stats, {}, {}

    # Compteur pour les fichiers traités
    files_processed = 0
    files_with_errors = 0
    
    # Parcourir tous les fichiers d'annotation dans le répertoire
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            txt_path = os.path.join(data_dir, file)
            
            try:
                with open(txt_path, 'r') as f:
                    lines_processed = 0
                    lines_with_errors = 0
                    
                    for line_num, line in enumerate(f, 1):
                        try:
                            # Diviser la ligne en parties
                            parts = line.strip().split()
                            
                            # Vérifier que la ligne contient suffisamment d'informations
                            if len(parts) < 5:
                                lines_with_errors += 1
                                logger.debug(f"Ligne {line_num} dans {file} : format incorrect (moins de 5 parties)")
                                continue

                            # Incrémenter le compteur total de bounding boxes
                            agreement_stats['total_boxes'] += 1
                            lines_processed += 1

                            # Extraire et valider les labels des experts (format: "x_y_z_w")
                            expert_labels_part = parts[0]
                            
                            # Vérifier le format des étiquettes d'experts
                            if expert_labels_part.count('_') != 3:
                                lines_with_errors += 1
                                logger.debug(f"Ligne {line_num} dans {file} : format d'étiquettes expert incorrect '{expert_labels_part}'")
                                continue
                            
                            # Diviser et convertir les étiquettes d'expert en entiers
                            try:
                                expert_labels = [int(label) for label in expert_labels_part.split('_')]
                                
                                # Vérifier que les étiquettes sont dans la plage valide (0-8 pour les classes de collemboles)
                                if any(label < 0 or label > 8 for label in expert_labels):
                                    lines_with_errors += 1
                                    logger.debug(f"Ligne {line_num} dans {file} : étiquette hors plage (0-8) '{expert_labels_part}'")
                                    continue
                                    
                            except ValueError:
                                lines_with_errors += 1
                                logger.debug(f"Ligne {line_num} dans {file} : étiquettes non numériques '{expert_labels_part}'")
                                continue

                            # Déterminer le pattern d'accord entre experts
                            # Cette fonction get_agreement_type classifie en "4/4", "3/1", "2/2", etc.
                            agreement_pattern = get_agreement_type(expert_labels)

                            # Calculer l'étiquette majoritaire en utilisant Counter pour compter les occurrences
                            label_counts = Counter(expert_labels)
                            majority_label = label_counts.most_common(1)[0][0]

                            # Mettre à jour les statistiques selon le type d'accord
                            if agreement_pattern == "4/4":
                                # Accord total - tous les experts sont d'accord
                                agreement_stats['full_agreement'] += 1
                                
                                # Mettre à jour la distribution des classes pour les cas d'accord total
                                # Ceci est précieux pour évaluer les classes les plus fiables
                                if majority_label not in class_distribution:
                                    class_distribution[majority_label] = 0
                                class_distribution[majority_label] += 1

                            elif agreement_pattern == "3/1":
                                # Accord majoritaire - 3 experts d'accord, 1 en désaccord
                                agreement_stats['majority_agreement'] += 1
                            else:
                                # Désaccord significatif (2/2, 2/1/1, 1/1/1/1)
                                agreement_stats['split_decision'] += 1

                            # Mettre à jour les statistiques d'accord par classe taxonomique
                            # Ceci permet d'identifier les classes plus difficiles à reconnaître
                            if majority_label not in agreement_by_class:
                                agreement_by_class[majority_label] = {
                                    'full': 0,      # Nombre d'accords totaux (4/4)
                                    'majority': 0,  # Nombre d'accords majoritaires (3/1)
                                    'split': 0,     # Nombre de désaccords significatifs
                                    'total': 0      # Nombre total d'annotations pour cette classe
                                }

                            agreement_by_class[majority_label]['total'] += 1

                            # Incrémenter le compteur approprié selon le type d'accord
                            if agreement_pattern == "4/4":
                                agreement_by_class[majority_label]['full'] += 1
                            elif agreement_pattern == "3/1":
                                agreement_by_class[majority_label]['majority'] += 1
                            else:
                                agreement_by_class[majority_label]['split'] += 1
                                
                        except Exception as e:
                            lines_with_errors += 1
                            logger.warning(f"Erreur lors du traitement de la ligne {line_num} dans {file}: {str(e)}")
                    
                    # Résumé du traitement du fichier
                    if lines_with_errors > 0:
                        logger.info(f"Fichier {file}: {lines_processed} lignes traitées, {lines_with_errors} lignes ignorées")
                
                files_processed += 1
                
            except Exception as e:
                files_with_errors += 1
                logger.error(f"Erreur lors de l'ouverture du fichier {txt_path}: {str(e)}")
    
    # Vérifier qu'il y a au moins une annotation
    total = agreement_stats['total_boxes']
    if total == 0:
        logger.warning("Aucune annotation trouvée dans les fichiers")
        return agreement_stats, {}, {}
    
    # Calculer les pourcentages d'accord
    # Ces pourcentages sont essentiels pour évaluer la cohérence globale des annotations
    agreement_stats['full_agreement_pct'] = agreement_stats['full_agreement'] / total * 100
    agreement_stats['majority_agreement_pct'] = agreement_stats['majority_agreement'] / total * 100
    agreement_stats['split_decision_pct'] = agreement_stats['split_decision'] / total * 100

    # Formater la distribution des classes en utilisant les noms lisibles
    # Cela facilite l'interprétation des résultats
    class_dist_formatted = {CLASS_MAPPING.get(k, f"Class {k}"): v for k, v in class_distribution.items()}

    # Formater les statistiques d'accord par classe avec les noms lisibles
    # et calculer les pourcentages pour chaque type d'accord
    agreement_by_class_formatted = {}
    for class_id, stats in agreement_by_class.items():
        class_name = CLASS_MAPPING.get(class_id, f"Class {class_id}")
        
        # Éviter la division par zéro
        total_class = stats['total']
        
        agreement_by_class_formatted[class_name] = {
            'full': stats['full'],
            'majority': stats['majority'],
            'split': stats['split'],
            'total': total_class,
            'full_pct': stats['full'] / total_class * 100 if total_class > 0 else 0,
            'majority_pct': stats['majority'] / total_class * 100 if total_class > 0 else 0,
            'split_pct': stats['split'] / total_class * 100 if total_class > 0 else 0
        }

    # Résumé du traitement global
    logger.info(f"Analyse d'accord terminée: {files_processed} fichiers traités, {files_with_errors} fichiers avec erreurs")
    logger.info(f"Total: {total} annotations, Accord total: {agreement_stats['full_agreement_pct']:.1f}%, "
               f"Accord majoritaire: {agreement_stats['majority_agreement_pct']:.1f}%")

    return agreement_stats, class_dist_formatted, agreement_by_class_formatted

def plot_agreement_stats(agreement_stats):
    """Affiche les statistiques d'accord entre experts"""
    labels = ['Full Agreement', 'Majority Agreement', 'Split Decision']
    values = [
        agreement_stats['full_agreement_pct'],
        agreement_stats['majority_agreement_pct'],
        agreement_stats['split_decision_pct']
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['green', 'blue', 'red'])
    plt.title('Expert Agreement Statistics')
    plt.xlabel('Agreement Type')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)

    for i, v in enumerate(values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agreement_stats.png'))
    plt.show()

def plot_class_distribution(class_distribution):
    """Affiche la distribution des classes"""
    plt.figure(figsize=(12, 6))

    classes = list(class_distribution.keys())
    counts = list(class_distribution.values())

    plt.bar(classes, counts)
    plt.title('Class Distribution (Full Agreement Cases)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    for i, v in enumerate(counts):
        plt.text(i, v + 5, str(v), ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.show()

def plot_agreement_by_class(agreement_by_class):
    """Visualise les statistiques d'accord pour chaque classe"""
    classes = list(agreement_by_class.keys())
    full_agreement = [stats['full_pct'] for stats in agreement_by_class.values()]
    majority_agreement = [stats['majority_pct'] for stats in agreement_by_class.values()]
    split_decision = [stats['split_pct'] for stats in agreement_by_class.values()]

    plt.figure(figsize=(14, 8))

    x = np.arange(len(classes))
    width = 0.25

    plt.bar(x - width, full_agreement, width, label='Full Agreement', color='green')
    plt.bar(x, majority_agreement, width, label='Majority Agreement', color='blue')
    plt.bar(x + width, split_decision, width, label='Split Decision', color='red')

    plt.title('Expert Agreement by Class')
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agreement_by_class.png'))
    plt.show()

def plot_agreement_performance(agreement_results):
    """Affiche les performances par type d'accord entre experts"""
    patterns = list(agreement_results.keys())
    accuracies = [agreement_results[p]['accuracy'] * 100 for p in patterns]
    counts = [agreement_results[p]['count'] for p in patterns]

    # Trier par difficulté croissante
    sorted_idx = np.argsort(accuracies)
    sorted_patterns = [patterns[i] for i in sorted_idx]
    sorted_accuracies = [accuracies[i] for i in sorted_idx]
    sorted_counts = [counts[i] for i in sorted_idx]

    # Créer une figure avec deux sous-graphes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique des précisions
    bars = ax1.bar(sorted_patterns, sorted_accuracies, color='skyblue')
    ax1.set_title('Performance par type d\'accord')
    ax1.set_xlabel('Pattern d\'accord')
    ax1.set_ylabel('Précision (%)')
    ax1.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{sorted_accuracies[i]:.1f}%',
                 ha='center', va='bottom')

    # Graphique des décomptes
    bars = ax2.bar(sorted_patterns, sorted_counts, color='salmon')
    ax2.set_title('Nombre d\'échantillons par type d\'accord')
    ax2.set_xlabel('Pattern d\'accord')
    ax2.set_ylabel('Nombre d\'échantillons')
    ax2.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{sorted_counts[i]}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agreement_performance.png'))
    plt.show()

def create_submission_file(predictions, output_file='submission.csv'):
    """Crée un fichier de soumission au format attendu par Kaggle"""
    with open(output_file, 'w') as f:
        f.write('idx,gt\n')
        for img_id, pred in predictions.items():
            f.write(f'{img_id},{pred}\n')

    print(f"Submission file created: {output_file}")

def track_memory_usage(model=None, phase=""):
    """
    Surveille l'utilisation de la mémoire CPU et GPU de manière optimisée,
    avec détails sur les composants du modèle si disponible.
    
    Args:
        model: Modèle PyTorch à analyser (optionnel)
        phase: Nom de la phase actuelle pour le logging
    """
    # Forcer le garbage collector avant les mesures
    gc.collect()
    if torch.cuda.is_available():
        # Vider le cache avant synchronisation pour des mesures plus précises
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Mesure CPU
    cpu_usage = psutil.virtual_memory()
    print(f"[{phase}] CPU Memory: {cpu_usage.used/(1024**3):.2f}/{cpu_usage.total/(1024**3):.2f} GB "
          f"({cpu_usage.percent:.1f}%)")
    
    if torch.cuda.is_available():
        try:
            # N'afficher que le GPU actuellement utilisé
            i = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
            max_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"[{phase}] GPU {i}: {allocated:.2f}/{max_mem:.2f} GB alloués ({allocated/max_mem*100:.1f}%), "
                  f"{cached:.2f} GB réservés, {cached-allocated:.2f} GB en cache")
        except Exception as e:
            print(f"[{phase}] Erreur lors de la mesure de mémoire GPU: {e}")
        
        # [...le reste de la fonction...]
        
        # Afficher les tailles des composants du modèle
        if model is not None:
            print("  Tailles approximatives des composants du modèle:")
            try:
                # Forcer la mesure sur CPU pour ne pas affecter les mesures GPU
                backbone_size = sum(p.numel() * p.element_size() for p in model.backbone.parameters()) / (1024 ** 3)
                print(f"  - Backbone: {backbone_size:.4f} GB")
                
                # FPN
                if hasattr(model, 'lateral_convs') and hasattr(model, 'fusion_convs'):
                    fpn_lat_size = sum(p.numel() * p.element_size() for p in model.lateral_convs.parameters()) / (1024 ** 3)
                    fpn_fusion_size = sum(p.numel() * p.element_size() for p in model.fusion_convs.parameters()) / (1024 ** 3)
                    print(f"  - FPN (lateral + fusion): {fpn_lat_size+fpn_fusion_size:.4f} GB")
                
                # Classifier
                if hasattr(model, 'classifier'):
                    classifier_size = sum(p.numel() * p.element_size() for p in model.classifier.parameters()) / (1024 ** 3)
                    print(f"  - Classifier: {classifier_size:.4f} GB")
                    
                # Buffers (BN stats, etc.)
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 ** 3)
                print(f"  - Buffers: {buffer_size:.4f} GB")
                
            except Exception as e:
                print(f"  Note: Erreur lors de l'analyse des composants: {e}")

#=======================================================================================
# FONCTION PRINCIPALE
#=======================================================================================

def main():
    # Définir le dispositif
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Nettoyage explicite de la mémoire GPU avant initialisation
    print("Nettoyage de la mémoire GPU avant initialisation du modèle...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()  # S'assurer que toutes les opérations GPU sont terminées
        print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} Go")
        print(f"Mémoire GPU utilisée: {torch.cuda.memory_allocated() / (1024**3):.2f} Go")
        
    # Surveillance de la mémoire au démarrage
    print_gpu_memory("démarrage")

    # 1. Traitement des données d'entraînement avec préservation des dimensions
    print("Processing training data with size preservation...")
    processor = CollemboleDataProcessor(train_dir)
    crops, labels, expert_labels_list, original_sizes, project_names, agreement_patterns, original_image_paths, original_bbox_coords = processor.crop_images()
    
    processor.visualize_and_save_background_samples(output_dir=output_dir)
    
    processor.visualize_samples(n_samples=3)

    # 2. Analyse de l'accord entre experts
    print("Analyzing expert agreement...")
    agreement_stats, class_distribution, agreement_by_class = analyze_expert_agreement(train_dir)
    print(f"Agreement statistics: {agreement_stats}")
    plot_agreement_stats(agreement_stats)
    plot_class_distribution(class_distribution)
    plot_agreement_by_class(agreement_by_class)

    # 3. Équilibrage adaptatif du jeu de données avec préservation des métadonnées
    print("Équilibrage adaptatif du jeu de données...")
    balanced_crops, balanced_labels, balanced_expert_labels, balanced_sizes, balanced_projects, balanced_patterns = balance_dataset(
        crops, labels, expert_labels_list, original_sizes, project_names, agreement_patterns,
        original_image_paths, original_bbox_coords, None, base_count=BASE_COUNT, augment=True
    ) 

    # 4. Préparation des transformations adaptées aux collemboles avec bruit gaussien
    print("Préparation des transformations avec bruit gaussien pour images microscopiques...")
    train_transform = get_collembole_transforms(training=True, target_size=INPUT_SIZE, with_noise=True)
    val_transform = get_collembole_transforms(training=False, target_size=INPUT_SIZE, with_noise=False)

    # Normalisation des tailles avec ratio d'aspect basée sur les données d'entraînement
    print("Initialisation du normaliseur de taille avec les dimensions des spécimens...")
    size_normalizer = SizeNormalizer(reference_sizes=balanced_sizes)
    print(f"Valeurs de référence - Largeur max: {size_normalizer.max_width:.2f}px, "
          f"Hauteur max: {size_normalizer.max_height:.2f}px, "
          f"Ratio max: {size_normalizer.max_ratio:.2f}")

    # 5. Préparation des données pour la validation croisée k-fold
    print("Préparation des données pour validation croisée k-fold...")

    # Créer le dataset complet - sans HOG pré-calculés
    full_dataset = CollemboleDataset(
        balanced_crops, balanced_labels, balanced_expert_labels, 
        balanced_sizes, balanced_projects, balanced_patterns,
        transform=train_transform, size_processor=size_normalizer
    )

    # Vérification de la présence de la classe FOND dans le dataset
    fond_count = sum(1 for label in balanced_labels if label == 8)
    print(f"Échantillons FOND: {fond_count} au total ({fond_count/len(balanced_labels)*100:.2f}%)")

    if fond_count == 0:
        print("ALERTE CRITIQUE: La classe FOND n'est pas présente dans le dataset!")
        print("Solution: Augmenter la génération d'échantillons de fond")

    # 6. Configuration de l'entraînement avec focal loss
    # Calcul des poids de classe inversement proportionnels à la fréquence
    class_counts = np.bincount(balanced_labels, minlength=len(CLASS_MAPPING))
    focal_loss = BalancedFocalLoss(class_counts, gamma=2.0, alpha=0.25)

    # Initialiser stratified k-fold pour préserver la distribution des classes dans chaque pli
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_histories = []
    all_val_metrics = []

    # Afficher la distribution des classes dans le dataset complet
    class_counts = np.bincount(balanced_labels)
    print("Distribution des classes dans le dataset complet:")
    for class_id, count in enumerate(class_counts):
        if count > 0:  # Afficher uniquement les classes présentes
            class_name = CLASS_MAPPING.get(class_id, f"Classe {class_id}")
            percentage = 100 * count / len(balanced_labels)
            print(f"  {class_name} (ID: {class_id}): {count} ({percentage:.1f}%)")

    # Utilisation de stratified k-fold avec les labels pour maintenir la proportion des classes
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), balanced_labels)):
        print(f"\n=== Entraînement du fold {fold+1}/5 ===")
        
        # Vérifier la distribution des classes dans ce fold
        train_labels = [balanced_labels[i] for i in train_idx]
        val_labels = [balanced_labels[i] for i in val_idx]
        
        train_class_counts = np.bincount(train_labels, minlength=len(CLASS_MAPPING))
        val_class_counts = np.bincount(val_labels, minlength=len(CLASS_MAPPING))
        
        print(f"Distribution des classes dans l'ensemble d'entraînement du fold {fold+1}:")
        for class_id, count in enumerate(train_class_counts):
            if count > 0:
                class_name = CLASS_MAPPING.get(class_id, f"Classe {class_id}")
                percentage = 100 * count / len(train_labels)
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"Distribution des classes dans l'ensemble de validation du fold {fold+1}:")
        for class_id, count in enumerate(val_class_counts):
            if count > 0:
                class_name = CLASS_MAPPING.get(class_id, f"Classe {class_id}")
                percentage = 100 * count / len(val_labels)
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Création des dataloaders pour ce fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_fn
        )

        # 7. Initialisation du modèle pour ce fold
        print(f"Initialisation du modèle pour le fold {fold+1}...")
        
        # Libérer explicitement la mémoire GPU avant initialisation
        print("Nettoyage de la mémoire GPU avant initialisation du modèle...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        # Choisir entre les backbones
        backbone_name = "resnet50"  # Alternative: "resnext50"

        # Initialisation d'abord sur CPU, puis transfert vers le device cible
        model = DynamicFPN(num_classes=len(CLASS_MAPPING), backbone_name=backbone_name, pretrained=True)
        model = model.to('cpu')  # Force l'initialisation sur CPU
        model = model.to(device)  # Transfert contrôlé vers le device final
        
        # Surveiller la mémoire après initialisation du modèle
        track_memory_usage(model, f"initialisation du modèle fold {fold+1}")

        # 8. Optimiseur et scheduler pour ce fold
        # Optimiseur AdamW avec paramètres optimisés
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=3e-4,
            weight_decay=0.05,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # OneCycleLR scheduler pour meilleure convergence et généralisation
        total_steps = len(train_loader) * N_EPOCHS
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0,
            anneal_strategy='cos'
        )

        # 9. Entraînement du modèle pour ce fold
        print(f"Entraînement du modèle fold {fold+1}...")
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=focal_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=N_EPOCHS,
            device=device,
            use_metadata=True
        )
        
        # Sauvegarder le modèle sur CPU
        model_path = os.path.join(output_dir, f'fold_{fold+1}_dynamicfpn_{backbone_name}.pth')
        torch.save(trained_model.cpu().state_dict(), model_path)
        fold_histories.append(history)
        
        # 10. Évaluation du modèle
        print(f"Évaluation du fold {fold+1}...")
        val_f1, val_f1_per_class, val_cm, val_uncertainty, agreement_results = evaluate_model(
            trained_model.to(device), val_loader, device, use_metadata=True, agreement_eval=True
        )
        
        # Stocker les résultats
        all_val_metrics.append({
            'f1': val_f1,
            'f1_per_class': val_f1_per_class,
            'cm': val_cm,
            'uncertainty': val_uncertainty,
            'agreement_results': agreement_results
        })
        
        # Afficher les résultats de ce fold
        print(f"Fold {fold+1} - F1 Score: {val_f1:.4f}, Uncertainty: {val_uncertainty:.4f}")
        
        # Nettoyer avant le prochain fold
        del trained_model, model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # 11. Synthèse des résultats de la validation croisée
    print("\n=== Synthèse des résultats k-fold ===")
    avg_f1 = np.mean([m['f1'] for m in all_val_metrics])
    avg_uncertainty = np.mean([m['uncertainty'] for m in all_val_metrics])
    print(f"F1 moyen sur les 5 folds: {avg_f1:.4f}")
    print(f"Incertitude moyenne sur les 5 folds: {avg_uncertainty:.4f}")

    # Calcul du F1 par classe sur tous les folds
    avg_f1_per_class = np.mean([m['f1_per_class'] for m in all_val_metrics], axis=0)
    print("F1 Score moyen par classe:")
    for class_id, score in enumerate(avg_f1_per_class):
        class_name = CLASS_MAPPING.get(class_id, f"Class {class_id}")
        print(f"  {class_name}: {score:.4f}")

    # Sélection du meilleur fold
    best_fold_idx = np.argmax([m['f1'] for m in all_val_metrics])
    best_fold_f1 = all_val_metrics[best_fold_idx]['f1']
    print(f"Meilleur fold: {best_fold_idx + 1} avec F1 = {best_fold_f1:.4f}")

    # Charger le meilleur modèle pour les prédictions sur le test
    print(f"Chargement du meilleur modèle (fold {best_fold_idx + 1}) pour les prédictions...")
    best_model = DynamicFPN(num_classes=len(CLASS_MAPPING), backbone_name=backbone_name, pretrained=False)
    best_model.load_state_dict(torch.load(os.path.join(output_dir, f'fold_{best_fold_idx + 1}_dynamicfpn_{backbone_name}.pth')))
    best_model = best_model.to(device)

    # Visualisation des résultats d'entraînement du meilleur fold
    plot_training_history(fold_histories[best_fold_idx])

    # Visualisation de la matrice de confusion du meilleur fold
    plot_confusion_matrix(all_val_metrics[best_fold_idx]['cm'], CLASS_MAPPING)

    # Visualisation des performances par type d'accord pour le meilleur fold
    plot_agreement_performance(all_val_metrics[best_fold_idx]['agreement_results'])

    # 12. Génération des prédictions sur les données de test avec approche par patchs optimisée
    print("Predicting test data using optimized patch-based approach...")

    # Préparation du dataset de test - sans transformations destructives
    test_dataset = TestDataset(test_dir, transform=None, size_normalizer=size_normalizer)

    # Prédiction directe image par image pour utiliser l'approche par patchs avec gestion mémoire optimisée
    predictions = {}
    all_probs = {}
    all_uncertainties = {}

    # Nouvelle boucle d'inférence optimisée
    for idx in tqdm(range(len(test_dataset)), desc="Processing test images"):
        image, image_id, original_size = test_dataset[idx]

        # Préprocessing direct
        img_tensor, metadata = dynamic_transform(image, add_metadata=True)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Métadonnées de taille
        if size_normalizer:
            w, h = image.size
            original_ratio = w / (h + 1e-8)
            size_features = size_normalizer.normalize((w, h, original_ratio))
            size_features = size_features.unsqueeze(0).to(device)
        else:
            w, h = image.size
            original_ratio = w / (h + 1e-8)
            size_features = torch.tensor([[w / 3000.0, h / 3000.0, original_ratio]], device=device)
        
        # Créer le masque de padding si nécessaire
        if metadata.get('padding_added', False):
            padding_offsets = metadata['padding_offsets']
            padding_mask = torch.zeros((1, 1, img_tensor.shape[2], img_tensor.shape[3]), device='cpu')
            new_w, new_h = metadata['resized_size']
            x_offset, y_offset = padding_offsets
            padding_mask[:, :, y_offset:y_offset+new_h, x_offset:x_offset+new_w] = 1
        else:
            padding_mask = torch.ones((1, 1, img_tensor.shape[2], img_tensor.shape[3]), device='cpu')
        
        # Transfert du masque au GPU
        padding_mask = padding_mask.to(device)
        
        # Prédiction directe avec le modèle
        with torch.no_grad():
            outputs, uncertainty = best_model(img_tensor, sizes=size_features, padding_mask=padding_mask)
        
        # Traitement des résultats (inchangé)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        
        # Stocker les résultats
        predictions[image_id] = pred
        all_probs[image_id] = probs.cpu().numpy().squeeze()
        all_uncertainties[image_id] = uncertainty.cpu().item()
        
        # Libérer la mémoire GPU
        del img_tensor, size_features, padding_mask, outputs, uncertainty
        if idx % 5 == 0:
            clear_gpu_memory()

    # 13. Créer le fichier de soumission
    create_submission_file(predictions, os.path.join(output_dir, f'submission_dynamicfpn_{backbone_name}.csv'))

    # 14. Analyser et visualiser les résultats de test
    print("\nTest prediction analysis:")

    # Distribution des classes prédites
    pred_classes = list(predictions.values())
    class_counts = Counter(pred_classes)
    print("Predicted class distribution:")
    for class_id, count in sorted(class_counts.items()):
        class_name = CLASS_MAPPING.get(class_id, f"Class {class_id}")
        percentage = 100 * count / len(predictions)
        print(f"  {class_name} (ID: {class_id}): {count} ({percentage:.1f}%)")

    # Analyse des incertitudes
    uncertainties = list(all_uncertainties.values())
    plt.figure(figsize=(10, 6))
    plt.hist(uncertainties, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(uncertainties), color='r', linestyle='--',
               label=f'Mean uncertainty: {np.mean(uncertainties):.3f}')
    plt.axvline(np.median(uncertainties), color='g', linestyle=':',
               label=f'Median uncertainty: {np.median(uncertainties):.3f}')
    plt.title('Distribution of prediction uncertainty for test images')
    plt.xlabel('Uncertainty')
    plt.ylabel('Number of images')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_uncertainty_distribution.png'))
    plt.show()

    # 15. Visualiser quelques exemples de prédictions avec Grad-CAM
    print("\nVisualizing sample predictions with Grad-CAM explanation...")

    # Sélectionner un échantillon d'images à visualiser (limiter pour économiser mémoire)
    sample_indices = random.sample(range(len(test_dataset)), min(3, len(test_dataset)))

    for idx in sample_indices:
        image, image_id, original_size = test_dataset[idx]
        pred_class = predictions[image_id]
        pred_prob = all_probs[image_id][pred_class]
        uncertainty = all_uncertainties[image_id]

        print(f"Image {image_id}:")
        print(f"  Predicted class: {CLASS_MAPPING.get(pred_class, f'Class {pred_class}')} (ID: {pred_class})")
        print(f"  Confidence: {pred_prob:.4f}")
        print(f"  Uncertainty: {uncertainty:.4f}")
        print(f"  Original size: {original_size}")

        # Redimensionner pour l'analyse Grad-CAM si l'image est très grande
        # tout en préservant le ratio d'aspect
        if max(original_size) > 1024:
            analysis_img = image.copy()
            ratio = 1024 / max(original_size)
            analysis_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            analysis_img = analysis_img.resize(analysis_size, Image.LANCZOS)
        else:
            analysis_img = image

        # Créer un tenseur pour l'analyse (sur CPU d'abord)
        img_tensor = dynamic_transform(analysis_img).unsqueeze(0)
        
        # Libérer la mémoire GPU avant analyse
        clear_gpu_memory()

        # Générer la carte de chaleur
        heatmap, _, _ = generate_heatmap(best_model, img_tensor, target_class=pred_class, device=device)
        
        if heatmap is not None:
            # Visualiser l'image avec la heatmap
            visualize_heatmap(analysis_img, heatmap, pred_class, pred_prob)

    # 16. Afficher les métriques finales et terminer
    print("\nFinal validation metrics:")
    print(f"  F1 Score: {val_f1:.4f}")
    print(f"  Uncertainty: {val_uncertainty:.4f}")

    print("\nTraining and evaluation complete. Models saved in directory: "
        f"{output_dir} with prefix fold_X_dynamicfpn_{backbone_name}.pth")
    print(f"Best model (fold {best_fold_idx + 1}) used for predictions.")
    print(f"Predictions saved as: submission_dynamicfpn_{backbone_name}.csv")

# Point d'entrée principal
if __name__ == "__main__":
    main()