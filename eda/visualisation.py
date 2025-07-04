import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from collections import Counter, defaultdict
from matplotlib.colors import rgb2hex
from sklearn.cluster import KMeans
import cv2
from pathlib import Path
import re


class ImageEDA:
    """
    Klasse für die explorative Datenanalyse von Bilddatensätzen.
    """

    def __init__(self, data_dir, sample_size=None, category_delimiter="_"):
        """
        Initialisiert die ImageEDA-Klasse.

        Args:
            data_dir (str): Verzeichnis mit Bilddaten
            sample_size (int, optional): Anzahl der für die Analyse zu verwendenden Bilder
            category_delimiter (str): Trennzeichen in Dateinamen zur Identifikation der Kategorie (z.B. '_' in 'Abyssinian_7.jpg')
        """
        self.data_dir = Path(data_dir)
        self.sample_size = sample_size
        self.category_delimiter = category_delimiter
        self.image_data = self._load_image_metadata()
        self.categories = self.image_data['category'].unique().tolist() if 'category' in self.image_data.columns else []

    def _load_image_metadata(self):
        """
        Lädt Metadaten über Bilder im Datensatz. Extrahiert Kategorien aus Dateinamen.
        
        Returns:
            pandas.DataFrame: DataFrame mit Bild-Metadaten
        """
        data = []
        
        # Finde alle Bilddateien
        image_files = (
            list(self.data_dir.glob("**/*.jpg")) +
            list(self.data_dir.glob("**/*.jpeg")) +
            list(self.data_dir.glob("**/*.png")) +
            list(self.data_dir.glob("**/*.bmp"))
        )
        
        if self.sample_size and len(image_files) > self.sample_size:
            image_files = random.sample(image_files, self.sample_size)
        
        for img_path in image_files:
            try:
                # Extrahiere Kategorie aus dem Dateinamen (z.B. "Abyssinian" aus "Abyssinian_7.jpg")
                filename = img_path.stem  # Name ohne Pfad und Erweiterung
                if self.category_delimiter in filename:
                    category = filename.split(self.category_delimiter)[0]
                else:
                    category = "unknown"

                with Image.open(img_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    file_size = img_path.stat().st_size / 1024  # KB

                    data.append(
                        {
                            "path": str(img_path),
                            "category": category,
                            "width": width,
                            "height": height,
                            "aspect_ratio": aspect_ratio,
                            "file_size": file_size,
                            "format": img.format,
                        }
                    )
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von {img_path}: {e}")

        return pd.DataFrame(data)

    def show_class_distribution_from_dirs(
        self,
        root_dir: str,
        figsize=(10, 6)
    ):
        """
        Zeigt die Verteilung der Bilder über grob gruppierte 'Klassen' an:
        - reine Zahlen werden zu 'numeric'
        - alle anderen Kategorien werden nach ihrem Präfix vor dem ersten '-'
            gruppiert (z.B. '2Q-1-' → '2Q', 'youtube-17' → 'youtube', 'Cats' bleibt 'Cats').
        """
        splits = ['train', 'valid', 'test']
        raw_counts = Counter()

        # Rohzählen pro Kategorie
        for split in splits:
            split_dir = os.path.join(root_dir, split)
            if not os.path.isdir(split_dir):
                continue
            pattern = os.path.join(split_dir, '*.*')
            for fp in glob.glob(pattern):
                fn = os.path.basename(fp)
                cat = fn.split('_', 1)[0] if '_' in fn else os.path.splitext(fn)[0]
                raw_counts[cat] += 1

        if not raw_counts:
            print("Keine Kategorien oder Bilder gefunden!")
            return

        # Gruppieren
        def _group(cat: str) -> str:
            # reine Zahlen
            if re.fullmatch(r'\d+', cat):
                return 'numeric'
            # 'irgendwas-' → 'irgendwas'
            if '-' in cat:
                return cat.split('-', 1)[0]
            # alles andere unverändert
            return cat

        grouped = defaultdict(int)
        for cat, cnt in raw_counts.items():
            grp = _group(cat)
            grouped[grp] += cnt

        # Plotten
        labels = list(grouped.keys())
        values = [grouped[l] for l in labels]

        plt.figure(figsize=figsize)
        sns.barplot(x=labels, y=values, palette="viridis")
        plt.title("Verteilung der Bilder nach Format­gruppen")
        plt.xlabel("Gruppe")
        plt.ylabel("Anzahl Bilder")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # 4) Konsole
        total_images = sum(values)
        print(f"Gesamtzahl der Gruppen: {len(grouped)}")
        print(f"Gesamtzahl der Bilder: {total_images}")
        print("Aufschlüsselung nach Gruppen:")
        for grp, cnt in sorted(grouped.items(), key=lambda x: -x[1]):
            print(f"  {grp}: {cnt}")

    def show_random_samples(self, samples_per_category=1, figsize=(12, 8)):
        """
        Zeigt zufällige Beispielbilder aus jeder Kategorie an.

        Args:
            samples_per_category (int): Anzahl der anzuzeigenden Beispiele pro Kategorie
            figsize (tuple): Größe der Abbildung
        """
        total_categories = len(self.categories)
        
        # Erstelle einzelne Subplots anstatt einer gemeinsamen Figur
        fig, axes = plt.subplots(
            nrows=total_categories, 
            ncols=samples_per_category, 
            figsize=figsize,
            constrained_layout=True  # Aktiviere constrained_layout
        )

        # Stelle sicher, dass axes immer 2D ist, auch bei einer Kategorie
        if total_categories == 1:
            axes = axes.reshape(1, -1)
        
        for i, category in enumerate(self.categories):
            # Hole Kategorieproben sicher
            category_data = self.image_data[self.image_data["category"] == category]
            sample_count = min(samples_per_category, len(category_data))
            
            if sample_count == 0:
                print(f"Warnung: Keine Proben für Kategorie '{category}' gefunden")
                for j in range(samples_per_category):
                    axes[i, j].text(0.5, 0.5, "Keine Beispiele vorhanden", 
                                ha='center', va='center')
                    axes[i, j].axis("off")
                continue
                
            category_samples = category_data["path"].sample(sample_count).tolist()
            
            # Fülle die verbleibenden Plätze mit leeren Plots, wenn nicht genug Beispiele
            for j in range(samples_per_category):
                ax = axes[i, j]
                if j < len(category_samples):
                    try:
                        img_path = category_samples[j]
                        img = Image.open(img_path)
                        img_res = img.resize((1920, 500))
                        ax.imshow(img_res)
                        
                        # Setze Titel über dem Bild mit mehr Abstand
                        ax.set_title(f"{category}", fontsize=10)
                        
                    except Exception as e:
                        print(f"Fehler beim Laden des Bildes {img_path}: {e}")
                        ax.text(0.5, 0.5, "Fehler beim Laden des Bildes", 
                            ha='center', va='center')
                else:
                    # Leer lassen, wenn nicht genug Beispiele
                    ax.set_visible(False)
                
                ax.axis("off")
        
        # Füge übergeordneten Titel hinzu
        plt.suptitle("Zufällige Beispiele nach Kategorie", fontsize=16, y=0.98)
        
        plt.show()

    def analyze_image_dimensions(self, figsize=(15, 10)):
        """
        Analysiert die Verteilung der Bildabmessungen und Seitenverhältnisse.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Breitenverteilung
        sns.histplot(self.image_data["width"], ax=axes[0, 0], bins=30)
        axes[0, 0].set_title("Breitenverteilung")
        axes[0, 0].set_xlabel("Breite (Pixel)")

        # Höhenverteilung
        sns.histplot(self.image_data["height"], ax=axes[0, 1], bins=30)
        axes[0, 1].set_title("Höhenverteilung")
        axes[0, 1].set_xlabel("Höhe (Pixel)")

        # Verteilung des Seitenverhältnisses
        sns.histplot(self.image_data["aspect_ratio"], ax=axes[1, 0], bins=30)
        axes[1, 0].set_title("Verteilung des Seitenverhältnisses")
        axes[1, 0].set_xlabel("Seitenverhältnis (Breite/Höhe)")

        # Verteilung der Dateigröße
        sns.histplot(self.image_data["file_size"], ax=axes[1, 1], bins=30)
        axes[1, 1].set_title("Verteilung der Dateigröße")
        axes[1, 1].set_xlabel("Dateigröße (KB)")

        plt.tight_layout()
        plt.show()

        # Zusammenfassende Statistiken ausgeben
        print("Statistiken der Bildabmessungen:")
        print(
            self.image_data[["width", "height", "aspect_ratio", "file_size"]].describe()
        )

    def analyze_colors(
        self,
        sample_size: int = 10,
        n_colors: int = 5,
        figsize=(12, 6)
    ):
        """
        Analysiert die dominanten Farben über alle Bilder im Datensatz.

        Args:
            sample_size (int): Anzahl der zufällig ausgewählten Bilder
            n_colors (int): Anzahl der zu extrahierenden dominanten Farben
            figsize (tuple): Größe der Figure
        """
        if self.image_data.empty:
            print("Keine Bilder zum Analysieren vorhanden!")
            return

        # Zufällige Teilmenge der Pfade
        paths = self.image_data["path"].sample(
            min(sample_size, len(self.image_data))
        ).tolist()

        # Pixel sammeln
        all_pixels = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (100, 100))
            all_pixels.append(img.reshape(-1, 3))
        if not all_pixels:
            print("Keine Pixel zum Clustern gefunden!")
            return
        all_pixels = np.vstack(all_pixels)

        # KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(all_pixels)
        colors = kmeans.cluster_centers_.astype(int)

        # Collage aus Beispielen
        collage = np.hstack([
            cv2.resize(
                cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
                (100, 100)
            )
            for p in paths[:5] if cv2.imread(p) is not None
        ])

        # Farbpalette
        palette = np.zeros((50, n_colors * 50, 3), dtype=np.uint8)
        for i, c in enumerate(colors):
            palette[:, i*50:(i+1)*50] = c

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(collage)
        ax1.set_title("Beispielbilder", fontsize=12)
        ax1.axis("off")

        ax2.imshow(palette)
        ax2.set_title("Dominante Farben", fontsize=12)
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    def show_augmentation_examples(self, category=None, index=0, figsize=(15, 10)):
        """
        Zeigt Beispiele häufiger Bildaugmentierungen an einem Beispielbild.
        """
        if 'category' not in self.image_data.columns or not self.categories:
            # Falls keine Kategorien vorhanden, zufälliges Bild nehmen
            random_image = self.image_data.sample(1)['path'].iloc[0]
            img_path = random_image
        else:
            if category is None:
                category = random.choice(self.categories)

        category_samples = self.image_data[self.image_data["category"] == category][
            "path"
        ].tolist()

        if not category_samples:
            print(f"Keine Beispiele für Kategorie {category} gefunden")
            return

        if index >= len(category_samples):
            index = 0

        img_path = category_samples[index]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Konnte Bild {img_path} nicht lesen")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Augmentierungen erstellen
        aug_flip_h = cv2.flip(img, 1)  # Horizontales Spiegeln
        aug_flip_v = cv2.flip(img, 0)  # Vertikales Spiegeln

        # Rotation
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
        aug_rotation = cv2.warpAffine(
            img, rotation_matrix, (img.shape[1], img.shape[0])
        )

        # Helligkeitsanpassung
        aug_bright = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        aug_dark = cv2.convertScaleAbs(img, alpha=0.5, beta=0)

        # Gaußscher Weichzeichner
        aug_blur = cv2.GaussianBlur(img, (15, 15), 0)

        # Subplots erstellen
        fig, axes = plt.subplots(2, 4, figsize=figsize)

        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(aug_flip_h)
        axes[0, 1].set_title("Horizontales Spiegeln")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(aug_flip_v)
        axes[0, 2].set_title("Vertikales Spiegeln")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(aug_rotation)
        axes[0, 3].set_title("Rotation (30°)")
        axes[0, 3].axis("off")

        axes[1, 0].imshow(aug_bright)
        axes[1, 0].set_title("Erhöhte Helligkeit")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(aug_dark)
        axes[1, 1].set_title("Verringerte Helligkeit")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(aug_blur)
        axes[1, 2].set_title("Gaußscher Weichzeichner")
        axes[1, 2].axis("off")

        axes[1, 3].axis("off")

        plt.suptitle(f"Augmentierungsbeispiele: {category}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self):
        """
        Führt einen vollständigen EDA-Workflow mit allen Visualisierungen durch.
        """
        print("=" * 50)
        print("STARTE EXPLORATIVE DATENANALYSE")
        print("=" * 50)

        print("\n1. Analyse der Klassenverteilung")
        self.show_class_distribution()

        print("\n2. Visualisierung zufälliger Beispiele")
        self.show_random_samples()

        print("\n3. Analyse der Bildabmessungen")
        self.analyze_image_dimensions()

        print("\n4. Farbanalyse nach Kategorie")
        self.analyze_colors_by_category()

        print("\n5. Augmentierungsbeispiele")
        self.show_augmentation_examples()

        print("\nEDA-Analyse abgeschlossen!")


# Anwendungsbeispiel
if __name__ == "__main__":
    # Aktuelles Arbeitsverzeichnis
    cwd = os.getcwd()

    # Datensatzpfad
    data_dir = os.path.join(cwd, 'data', 'images')

    # EDA-Tool initialisieren
    eda = ImageEDA(data_dir, category_delimiter="_")

    # Vollständige Analyse durchführen
    eda.run_complete_analysis()

    # Oder einzelne Analysen ausführen
    # eda.show_class_distribution()
    # eda.show_random_samples(samples_per_category=3)
    # eda.analyze_image_dimensions()
    # eda.analyze_colors_by_category(samples_per_category=5, n_colors=3)
    # eda.show_augmentation_examples()