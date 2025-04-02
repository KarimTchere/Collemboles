import os
import json
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np

class CompleteCollemboleAnnotator:
    def __init__(self):
        # Définir les espèces de collemboles
        self.species = [
            "AUTRE",    # 0
            "Cer",      # 1
            "CRY_THE",  # 2
            "HYP_MAN",  # 3
            "ISO_MIN",  # 4
            "LEP",      # 5
            "MET_AFF",  # 6
            "PAR_NOT",  # 7
            "FOND"      # 8
        ]
        
        # Définir les caractéristiques
        self.features = [
            "Antennes",
            "Furca",
            "Corps segmenté",
            "Tête distincte",
            "Couleur claire",
            "Couleur foncée",
            "Poils visibles",
            "Forme allongée",
            "Forme globuleuse",
            "Appendice",
            'Yeux',
            'Queue'
        ]
        
        # Fichier pour sauvegarder les annotations
        self.annotation_file = "collembole_annotationffs.json"
        
        # Structure pour stocker les annotations
        self.annotations = {}
        
        # Charger les annotations existantes si disponibles
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r') as f:
                    self.annotations = json.load(f)
                print(f"Annotations chargées: {len(self.annotations)} images.")
            except:
                print("Impossible de charger les annotations existantes.")

        # Variables pour stocker l'état actuel
        self.current_image_path = None
        self.current_image_name = None
        self.current_species = None
        self.current_species_idx = -1
        self.current_bbox = None  # Pour stocker les coordonnées de la bounding box
        
        # Variables pour le dessin de bounding boxes
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.drawing_bbox = False
        
        # Créer la fenêtre principale
        self.root = tk.Tk()
        self.root.title("Annotateur Complet de Collemboles")
        self.root.geometry("1200x800")
        
        # Créer le cadre principal
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Cadre pour les informations et les contrôles
        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Cadre pour l'image
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas pour afficher l'image
        self.canvas = tk.Canvas(self.image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Afficher l'information sur l'espèce avec la possibilité de la changer
        self.species_frame = tk.Frame(self.info_frame)
        self.species_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(self.species_frame, text="Espèce:").pack(anchor=tk.W)
        
        # Menu déroulant pour sélectionner l'espèce
        self.species_var = tk.StringVar(self.root)
        self.species_dropdown = tk.OptionMenu(self.species_frame, self.species_var, *self.species)
        self.species_dropdown.pack(fill=tk.X, pady=5)
        self.species_var.set(self.species[0])  # Valeur par défaut
        
        # Quand l'espèce est changée
        def on_species_change(*args):
            if self.current_image_name:
                self.current_species = self.species_var.get()
                self.current_species_idx = self.species.index(self.current_species)
                
                # Mettre à jour les annotations
                if self.current_image_name in self.annotations:
                    self.annotations[self.current_image_name]["species"] = self.current_species
                    self.annotations[self.current_image_name]["species_idx"] = self.current_species_idx
                    self.save_annotations()
        
        # Lier la fonction au changement de variable
        self.species_var.trace_add("write", on_species_change)
        
        # Afficher les informations de bounding box
        tk.Label(self.info_frame, text="Bounding Box:").pack(anchor=tk.W, pady=5)
        self.bbox_info = tk.Label(self.info_frame, text="Aucune information", justify=tk.LEFT)
        self.bbox_info.pack(anchor=tk.W, pady=5)
        
        # Afficher le contenu brut du fichier TXT
        tk.Label(self.info_frame, text="Fichier TXT:").pack(anchor=tk.W, pady=5)
        self.txt_content = tk.Label(self.info_frame, text="Pas de fichier chargé", 
                                  justify=tk.LEFT, wraplength=250)
        self.txt_content.pack(anchor=tk.W, pady=5)
        
        # Boutons pour la navigation et l'export
        self.button_frame = tk.Frame(self.info_frame)
        self.button_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(self.button_frame, text="Ouvrir un dossier", command=self.open_folder).pack(fill=tk.X, pady=2)
        tk.Button(self.button_frame, text="Image précédente", command=self.prev_image).pack(fill=tk.X, pady=2)
        tk.Button(self.button_frame, text="Image suivante", command=self.next_image).pack(fill=tk.X, pady=2)
        tk.Button(self.button_frame, text="Exporter JSON", command=self.export_json).pack(fill=tk.X, pady=2)
        tk.Button(self.button_frame, text="Ajouter caractéristique", command=self.add_feature_to_current).pack(fill=tk.X, pady=2)
        
        # Options pour le mode d'affichage
        self.view_frame = tk.Frame(self.info_frame)
        self.view_frame.pack(fill=tk.X, pady=5)
        
        # Option pour montrer l'image recadrée
        self.crop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.view_frame, text="Montrer l'image recadrée", 
                      variable=self.crop_var, command=self.toggle_crop_view).pack(anchor=tk.W)
        
        # Option pour activer/désactiver le dessin de bounding boxes
        self.draw_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.view_frame, text="Dessiner bounding box", 
                     variable=self.draw_var).pack(anchor=tk.W)
        
        # Zone de texte pour les instructions
        tk.Label(self.info_frame, text="Instructions:").pack(anchor=tk.W, pady=(10, 0))
        
        self.instructions = tk.Text(self.info_frame, height=8, width=30)
        self.instructions.pack(fill=tk.X, pady=5)
        self.instructions.insert(tk.END, "1. Ouvrez un dossier d'images\n2. Choisissez l'espèce\n3. L'image est affichée avec la bounding box du TXT\n4. Pour dessiner une nouvelle bounding box:\n   - Cochez 'Dessiner bounding box'\n   - Cliquez-glissez sur l'image\n5. Ajoutez des caractéristiques avec le bouton")
        self.instructions.config(state=tk.DISABLED)
        
        # Liste des annotations
        tk.Label(self.info_frame, text="Caractéristiques:").pack(anchor=tk.W, pady=(10, 0))
        
        self.feature_listbox = tk.Listbox(self.info_frame, height=10)
        self.feature_listbox.pack(fill=tk.X, pady=5)
        
        # Bouton pour supprimer une caractéristique
        tk.Button(self.info_frame, text="Supprimer la caractéristique sélectionnée", 
                 command=self.delete_selected_feature).pack(fill=tk.X, pady=2)
        
        # Binding des événements pour le dessin
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Initialiser les variables
        self.image_paths = []
        self.current_image_index = -1
        self.original_img = None  # Pour stocker l'image originale
        self.img_scale = 1.0  # Facteur d'échelle pour l'image
        
        # Démarrer l'application
        self.root.mainloop()
    
    def toggle_crop_view(self):
        """Bascule entre l'affichage de l'image complète et l'image recadrée"""
        if self.current_image_path:
            self.show_current_image()
    
    def on_press(self, event):
        """Gère l'événement de clic pour dessiner une bounding box"""
        if not self.draw_var.get() or self.crop_var.get():
            return  # Ne pas dessiner si le mode dessin n'est pas activé ou en mode crop
        
        self.start_x = event.x
        self.start_y = event.y
        self.drawing_bbox = True
        
        # Créer un rectangle initial
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="blue", width=2
        )
    
    def on_drag(self, event):
        """Gère l'événement de glissement pour dessiner une bounding box"""
        if not self.drawing_bbox:
            return
        
        # Mettre à jour le rectangle
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)
        
        # Afficher les coordonnées temporaires
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Convertir les coordonnées du canvas en coordonnées normalisées
        if self.original_img:
            img_width, img_height = self.original_img.size
            
            # Calculer le rectangle en pixels
            x1 = min(self.start_x, event.x) / self.img_scale
            y1 = min(self.start_y, event.y) / self.img_scale
            x2 = max(self.start_x, event.x) / self.img_scale
            y2 = max(self.start_y, event.y) / self.img_scale
            
            # Calculer le centre et les dimensions en format YOLO
            center_x = (x1 + x2) / 2 / canvas_width
            center_y = (y1 + y2) / 2 / canvas_height
            width = abs(x2 - x1) / canvas_width
            height = abs(y2 - y1) / canvas_height
            
            # Mettre à jour l'affichage des coordonnées
            bbox_text = f"Centre X: {center_x:.4f}, Centre Y: {center_y:.4f}\nLargeur: {width:.4f}, Hauteur: {height:.4f}"
            self.bbox_info.config(text=bbox_text)
    
    def on_release(self, event):
        """Gère l'événement de relâchement pour finaliser une bounding box"""
        if not self.drawing_bbox:
            return
        
        # Réinitialiser l'état de dessin
        self.drawing_bbox = False
        
        # Obtenir les dimensions du canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculer le rectangle en pixels
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Ignorer les rectangles trop petits
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            return
        
        # Calculer le centre et les dimensions en format YOLO
        center_x = (x1 + x2) / 2 / canvas_width
        center_y = (y1 + y2) / 2 / canvas_height
        width = abs(x2 - x1) / canvas_width
        height = abs(y2 - y1) / canvas_height
        
        # Mettre à jour la bounding box actuelle
        self.current_bbox = {
            "center_x": center_x,
            "center_y": center_y,
            "width": width,
            "height": height
        }
        
        # Afficher un point au centre de la nouvelle bounding box
        self.canvas.create_oval(
            (x1 + x2) / 2 - 3, (y1 + y2) / 2 - 3,
            (x1 + x2) / 2 + 3, (y1 + y2) / 2 + 3,
            fill="blue", outline="blue"
        )
        
        # Demander la caractéristique pour cette nouvelle bounding box
        self.add_feature_to_current()
    
    def add_new_feature(self):
        """Ajoute une nouvelle caractéristique à la liste"""
        new_feature = simpledialog.askstring("Nouvelle caractéristique", 
                                            "Entrez le nom de la nouvelle caractéristique:",
                                            parent=self.root)
        
        # Vérifier que la caractéristique est valide
        if new_feature and new_feature.strip() and new_feature not in self.features:
            # Ajouter la caractéristique à la liste
            self.features.append(new_feature)
            messagebox.showinfo("Succès", f"Caractéristique '{new_feature}' ajoutée avec succès!")
            return new_feature
        return None
    
    def add_feature_to_current(self):
        """Ajoute une caractéristique à l'image courante"""
        if not self.current_image_path:
            messagebox.showinfo("Information", "Aucune image n'est affichée.")
            return
        
        if not self.current_bbox:
            messagebox.showinfo("Information", "Aucune bounding box sélectionnée.")
            return
        
        # Demander la caractéristique
        feature_window = tk.Toplevel(self.root)
        feature_window.title("Ajouter une caractéristique")
        feature_window.geometry("300x400")
        feature_window.transient(self.root)
        feature_window.grab_set()
        
        tk.Label(feature_window, text="Sélectionnez une caractéristique:").pack(pady=5)
        
        # Menu déroulant pour les caractéristiques existantes
        feature_var = tk.StringVar(feature_window)
        feature_dropdown = tk.OptionMenu(feature_window, feature_var, *self.features)
        feature_dropdown.pack(fill=tk.X, padx=10, pady=5)
        
        if self.features:
            feature_var.set(self.features[0])
        
        # Option pour ajouter une nouvelle caractéristique
        def add_new():
            new_feature = self.add_new_feature()
            if new_feature:
                # Mettre à jour le menu déroulant
                menu = feature_dropdown["menu"]
                menu.delete(0, "end")
                for feature in self.features:
                    menu.add_command(label=feature, 
                                    command=lambda value=feature: feature_var.set(value))
                feature_var.set(new_feature)
        
        tk.Button(feature_window, text="Nouvelle caractéristique", command=add_new).pack(pady=5)
        
        tk.Label(feature_window, text="Description (optionnelle):").pack(pady=5)
        description_entry = tk.Entry(feature_window)
        description_entry.pack(fill=tk.X, padx=10, pady=5)
        
        # Fonction pour valider la sélection
        def validate():
            # Obtenir la caractéristique
            feature_name = feature_var.get()
            
            # Obtenir la description
            description = description_entry.get().strip()
            
            # Ajouter la caractéristique à l'image courante
            if self.current_image_name not in self.annotations:
                self.annotations[self.current_image_name] = {
                    "path": self.current_image_path,
                    "species": self.current_species,
                    "species_idx": self.current_species_idx,
                    "features": {}
                }
            
            if "features" not in self.annotations[self.current_image_name]:
                self.annotations[self.current_image_name]["features"] = {}
            
            self.annotations[self.current_image_name]["features"][feature_name] = {
                "description": description,
                "bbox": self.current_bbox  # Ajouter les coordonnées de la bounding box
            }
            
            # Sauvegarder les annotations
            self.save_annotations()
            
            # Mettre à jour l'affichage
            self.update_feature_listbox()
            
            # Fermer la fenêtre
            feature_window.destroy()
            
            messagebox.showinfo("Succès", f"Caractéristique '{feature_name}' ajoutée à l'image.")
            
            # Réinitialiser le rectangle de dessin
            if self.rect_id:
                self.canvas.itemconfig(self.rect_id, outline="green")  # Changer la couleur pour indiquer que c'est annoté
        
        # Bouton de validation
        tk.Button(feature_window, text="Ajouter", command=validate).pack(pady=10)
        
        # Centrer la fenêtre
        feature_window.update_idletasks()
        width = feature_window.winfo_width()
        height = feature_window.winfo_height()
        x = (feature_window.winfo_screenwidth() // 2) - (width // 2)
        y = (feature_window.winfo_screenheight() // 2) - (height // 2)
        feature_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Attendre que la fenêtre soit fermée
        self.root.wait_window(feature_window)
    
    def delete_selected_feature(self):
        """Supprime la caractéristique sélectionnée"""
        if not self.feature_listbox.curselection():
            messagebox.showinfo("Information", "Aucune caractéristique sélectionnée.")
            return
        
        # Obtenir la caractéristique sélectionnée
        index = self.feature_listbox.curselection()[0]
        feature_name = self.feature_listbox.get(index).split(": ")[0]
        
        # Confirmer la suppression
        confirm = messagebox.askyesno("Confirmation", 
                                     f"Voulez-vous vraiment supprimer la caractéristique '{feature_name}'?")
        if not confirm:
            return
        
        # Supprimer la caractéristique
        if self.current_image_name in self.annotations:
            if "features" in self.annotations[self.current_image_name]:
                if feature_name in self.annotations[self.current_image_name]["features"]:
                    del self.annotations[self.current_image_name]["features"][feature_name]
                    
                    # Sauvegarder les annotations
                    self.save_annotations()
                    
                    # Mettre à jour l'affichage
                    self.update_feature_listbox()
                    
                    messagebox.showinfo("Succès", f"Caractéristique '{feature_name}' supprimée.")
    
    def open_folder(self):
        """Ouvre un dossier d'images"""
        folder_path = filedialog.askdirectory(title="Sélectionnez un dossier d'images")
        
        if not folder_path:
            return
        
        # Recherche des images
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        self.image_paths = []
        
        for file in os.listdir(folder_path):
            base_name, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                # Vérifier s'il existe un fichier texte associé
                txt_file = os.path.join(folder_path, base_name + ".txt")
                if os.path.exists(txt_file):
                    # Vérifier que le fichier TXT ne contient qu'une seule ligne
                    with open(txt_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) == 1:
                            self.image_paths.append(os.path.join(folder_path, file))
        
        if not self.image_paths:
            messagebox.showinfo("Information", "Aucune image valide trouvée dans ce dossier")
            return
        
        messagebox.showinfo("Information", f"{len(self.image_paths)} images valides trouvées")
        
        # Afficher la première image
        self.current_image_index = 0
        self.show_current_image()
    
    def show_current_image(self):
        """Affiche l'image actuelle avec les informations du TXT"""
        if not self.image_paths or self.current_image_index < 0 or self.current_image_index >= len(self.image_paths):
            return
        
        # Nettoyer le canvas
        self.canvas.delete("all")
        
        # Charger l'image
        image_path = self.image_paths[self.current_image_index]
        self.current_image_path = image_path
        self.current_image_name = os.path.basename(image_path)
        
        # Mettre à jour le titre de la fenêtre
        self.root.title(f"Annotation - {self.current_image_name} ({self.current_image_index + 1}/{len(self.image_paths)})")
        
        # Lire le fichier texte associé
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        
        try:
            # Lire le contenu du fichier TXT
            with open(txt_path, 'r') as f:
                line = f.readline().strip()
                
                # Afficher le contenu brut du fichier TXT
                self.txt_content.config(text=line)
                
                # Diviser la ligne en parties
                parts = line.split()
                
                if len(parts) >= 4:  # Au moins 4 éléments sont nécessaires
                    # Extraire l'espèce (premier élément)
                    label_parts = parts[0].split('_')
                    if len(label_parts) >= 1 and label_parts[0].isdigit():
                        self.current_species_idx = int(label_parts[0])
                        
                        if 0 <= self.current_species_idx < len(self.species):
                            self.current_species = self.species[self.current_species_idx]
                            self.species_var.set(self.current_species)
                        else:
                            self.current_species = f"Inconnu ({self.current_species_idx})"
                            self.species_var.set(self.species[0])
                    else:
                        self.current_species_idx = -1
                        self.current_species = "Inconnu"
                        self.species_var.set(self.species[0])
                    
                    # Extraire les coordonnées (les 4 derniers éléments)
                    # Format YOLO: x_center y_center width height (valeurs normalisées)
                    center_x = float(parts[-4])
                    center_y = float(parts[-3])
                    width = float(parts[-2])
                    height = float(parts[-1])
                    
                    # Stocker les coordonnées
                    self.current_bbox = {
                        "center_x": center_x,
                        "center_y": center_y,
                        "width": width,
                        "height": height
                    }
                    
                    # Mettre à jour l'affichage des coordonnées
                    bbox_text = f"Centre X: {center_x:.4f}, Centre Y: {center_y:.4f}\nLargeur: {width:.4f}, Hauteur: {height:.4f}"
                    self.bbox_info.config(text=bbox_text)
                    
                    # Charger l'image complète
                    try:
                        full_img = Image.open(image_path)
                        self.original_img = full_img
                        
                        # Calculer les coordonnées en pixels pour la bounding box
                        img_width, img_height = full_img.size
                        pixel_center_x = int(center_x * img_width)
                        pixel_center_y = int(center_y * img_height)
                        pixel_width = int(width * img_width)
                        pixel_height = int(height * img_height)
                        
                        # Calculer les coordonnées du coin supérieur gauche
                        pixel_x = pixel_center_x - pixel_width // 2
                        pixel_y = pixel_center_y - pixel_height // 2
                        
                        # Si l'option "Montrer l'image recadrée" est activée
                        if self.crop_var.get():
                            # Rogner l'image
                            cropped_img = full_img.crop((
                                max(0, pixel_x),
                                max(0, pixel_y),
                                min(img_width, pixel_x + pixel_width),
                                min(img_height, pixel_y + pixel_height)
                            ))
                            
                            # Redimensionner l'image rognée
                            canvas_width = self.canvas.winfo_width()
                            canvas_height = self.canvas.winfo_height()
                            
                            if canvas_width <= 1 or canvas_height <= 1:
                                canvas_width = 800
                                canvas_height = 600
                            
                            cropped_width, cropped_height = cropped_img.size
                            scale_x = canvas_width / cropped_width
                            scale_y = canvas_height / cropped_height
                            scale = min(scale_x, scale_y)
                            
                            new_width = int(cropped_width * scale)
                            new_height = int(cropped_height * scale)
                            
                            resized_img = cropped_img.resize((new_width, new_height), Image.LANCZOS)
                            
                            # Afficher l'image rognée
                            self.tk_img = ImageTk.PhotoImage(resized_img)
                            self.canvas.config(width=new_width, height=new_height)
                            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
                            
                            # Dans ce mode, on ne peut pas dessiner de bounding boxes
                            self.draw_var.set(False)
                        else:
                            # Redimensionner l'image complète pour le canvas
                            canvas_width = self.canvas.winfo_width()
                            canvas_height = self.canvas.winfo_height()
                            
                            if canvas_width <= 1 or canvas_height <= 1:
                                canvas_width = 800
                                canvas_height = 600
                            
                            scale_x = canvas_width / img_width
                            scale_y = canvas_height / img_height
                            scale = min(scale_x, scale_y)
                            
                            new_width = int(img_width * scale)
                            new_height = int(img_height * scale)
                            
                            # Stocker le facteur d'échelle pour le dessin
                            self.img_scale = scale
                            
                            resized_img = full_img.resize((new_width, new_height), Image.LANCZOS)
                            
                            # Afficher l'image complète
                            self.tk_img = ImageTk.PhotoImage(resized_img)
                            self.canvas.config(width=new_width, height=new_height)
                            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
                            
                            # Dessiner un rectangle pour la bounding box
                            # Convertir les coordonnées de la bounding box pour l'image redimensionnée
                            scaled_center_x = center_x * new_width
                            scaled_center_y = center_y * new_height
                            scaled_width = width * new_width
                            scaled_height = height * new_height
                            
                            # Calculer les coins du rectangle
                            left = scaled_center_x - scaled_width / 2
                            top = scaled_center_y - scaled_height / 2
                            right = scaled_center_x + scaled_width / 2
                            bottom = scaled_center_y + scaled_height / 2
                            
                            self.canvas.create_rectangle(
                                left, top, right, bottom,
                                outline="red", width=2
                            )
                            
                            # Afficher un point au centre de la bounding box
                            self.canvas.create_oval(
                                scaled_center_x - 3, scaled_center_y - 3,
                                scaled_center_x + 3, scaled_center_y + 3,
                                fill="red", outline="red"
                            )
                            
                            # Ajouter un texte explicatif sur le rectangle
                            self.canvas.create_text(
                                left, top - 10,
                                text=f"TXT: {self.species[self.current_species_idx]}",
                                fill="red", anchor=tk.W
                            )
                        
                        # Mettre à jour la liste des caractéristiques
                        self.update_feature_listbox()
                        
                        # Dessiner les bounding boxes des caractéristiques
                        self.draw_feature_boxes()
                    except Exception as e:
                        print(f"Erreur lors de l'affichage de l'image: {e}")
                        messagebox.showerror("Erreur", f"Impossible d'afficher l'image: {e}")
                else:
                    messagebox.showerror("Erreur", f"Format de fichier TXT invalide (moins de 4 éléments): {txt_path}")
                    self.next_image()
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de traiter l'image: {e}")
            self.next_image()
    
    def draw_feature_boxes(self):
        """Dessine les bounding boxes des caractéristiques annotées"""
        if self.crop_var.get():
            return  # Ne pas dessiner en mode recadré
        
        if self.current_image_name not in self.annotations:
            return
        
        if "features" not in self.annotations[self.current_image_name]:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        for feature, details in self.annotations[self.current_image_name]["features"].items():
            if "bbox" in details and details["bbox"]:
                bbox = details["bbox"]
                
                if "center_x" in bbox and "center_y" in bbox and "width" in bbox and "height" in bbox:
                    # Convertir les coordonnées normalisées en pixels pour le canvas
                    center_x = bbox["center_x"] * canvas_width
                    center_y = bbox["center_y"] * canvas_height
                    width = bbox["width"] * canvas_width
                    height = bbox["height"] * canvas_height
                    
                    # Calculer les coins du rectangle
                    left = center_x - width / 2
                    top = center_y - height / 2
                    right = center_x + width / 2
                    bottom = center_y + height / 2
                    
                    # Dessiner le rectangle
                    self.canvas.create_rectangle(
                        left, top, right, bottom,
                        outline="green", width=2
                    )
                    
                    # Afficher le nom de la caractéristique
                    self.canvas.create_text(
                        left, top - 10,
                        text=feature,
                        fill="green", anchor=tk.W
                    )
    
    def update_feature_listbox(self):
        """Met à jour la liste des caractéristiques"""
        self.feature_listbox.delete(0, tk.END)
        
        if self.current_image_name not in self.annotations:
            return
        
        # Afficher toutes les caractéristiques
        if "features" in self.annotations[self.current_image_name]:
            for feature, details in self.annotations[self.current_image_name]["features"].items():
                description = details.get("description", "")
                display_text = f"{feature}: {description}"
                self.feature_listbox.insert(tk.END, display_text)
    
    def prev_image(self):
        """Affiche l'image précédente"""
        if not self.image_paths:
            return
        
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()
    
    def next_image(self):
        """Affiche l'image suivante"""
        if not self.image_paths:
            return
        
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.show_current_image()
    
    def save_annotations(self):
        """Sauvegarde les annotations dans un fichier JSON"""
        with open(self.annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print("Annotations sauvegardées.")
    
    def export_json(self):
        """Exporte les annotations au format JSON dans un fichier séparé"""
        if not self.annotations:
            messagebox.showinfo("Information", "Aucune annotation à exporter.")
            return
        
        # Demander le nom du fichier
        output_file = filedialog.asksaveasfilename(
            title="Enregistrer les annotations",
            defaultextension=".json",
            filetypes=[("Fichier JSON", "*.json")]
        )
        
        if not output_file:
            return
        
        # Créer un dictionnaire au format demandé
        export_data = {}
        
        for image_name, annotation in self.annotations.items():
            if "features" not in annotation:
                continue
            
            image_data = {
                "species": annotation.get("species", "Inconnue"),
                "species_idx": annotation.get("species_idx", -1),
                "bbox": annotation.get("bbox", {}),
                "features": {}
            }
            
            for feature_name, details in annotation["features"].items():
                image_data["features"][feature_name] = {
                    "description": details.get("description", ""),
                    "bbox": details.get("bbox", {})
                }
            
            export_data[image_name] = image_data
        
        # Sauvegarder le fichier
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        messagebox.showinfo("Information", f"Annotations exportées dans {output_file}")

if __name__ == "__main__":
    app = CompleteCollemboleAnnotator()