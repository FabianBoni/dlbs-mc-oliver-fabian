import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from collections import Counter
from matplotlib.colors import rgb2hex
from sklearn.cluster import KMeans
import cv2
from pathlib import Path


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

    def show_class_distribution(self, figsize=(10, 6)):
        """
        Zeigt die Verteilung der Bilder über Kategorien hinweg an.
        """
        if 'category' not in self.image_data.columns or self.image_data.empty:
            print("Keine Kategorien gefunden oder Datensatz ist leer!")
            return
        plt.figure(figsize=figsize)
        counts = self.image_data["category"].value_counts()
        sns.barplot(x=counts.index, y=counts.values)
        plt.title("Verteilung der Bilder über Kategorien")
        plt.xlabel("Kategorie")
        plt.ylabel("Anzahl der Bilder")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print(f"Gesamtzahl der Kategorien: {len(counts)}")
        print(f"Gesamtzahl der Bilder: {len(self.image_data)}")
        print(f"Aufschlüsselung nach Kategorien:\n{counts}")

    def show_random_samples(self, samples_per_category=5, figsize=(15, 10)):
        """
        Zeigt zufällige Beispielbilder aus jeder Kategorie an.

        Args:
            samples_per_category (int): Anzahl der anzuzeigenden Beispiele pro Kategorie
            figsize (tuple): Größe der Abbildung
        """
        total_categories = len(self.categories)
        fig, axes = plt.subplots(
            total_categories, samples_per_category, figsize=figsize
        )

        if total_categories == 1:
            axes = [axes]

        for i, category in enumerate(self.categories):
            category_samples = (
                self.image_data[self.image_data["category"] == category]["path"]
                .sample(
                    min(
                        samples_per_category,
                        len(self.image_data[self.image_data["category"] == category]),
                    )
                )
                .tolist()
            )

            for j, img_path in enumerate(category_samples):
                if j < samples_per_category:  # Sicherheitsüberprüfung
                    img = plt.imread(img_path)
                    if total_categories == 1:
                        ax = axes[j]
                    else:
                        ax = axes[i, j]
                    ax.imshow(img)
                    ax.set_title(f"{category}")
                    ax.axis("off")

        plt.tight_layout()
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

    def analyze_colors_by_category(
        self, samples_per_category=10, n_colors=5, figsize=(15, 10)
    ):
        """
        Analysiert die dominanten Farben für jede Kategorie.

        Args:
            samples_per_category (int): Anzahl der Beispiele pro Kategorie
            n_colors (int): Anzahl der zu extrahierenden dominanten Farben
            figsize (tuple): Größe der Abbildung
        """
        fig, axes = plt.subplots(len(self.categories), 2, figsize=figsize)

        if len(self.categories) == 1:
            axes = [axes]

        for i, category in enumerate(self.categories):
            category_samples = (
                self.image_data[self.image_data["category"] == category]["path"]
                .sample(
                    min(
                        samples_per_category,
                        len(self.image_data[self.image_data["category"] == category]),
                    )
                )
                .tolist()
            )

            if not category_samples:
                continue

            # Farbinformationen aus allen Beispielen kombinieren
            all_pixels = []
            for img_path in category_samples:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(
                    img, (100, 100)
                )  # Verkleinern für schnellere Verarbeitung
                pixels = img.reshape(-1, 3)
                all_pixels.append(pixels)

            if not all_pixels:
                continue

            all_pixels = np.vstack(all_pixels)

            # K-Means verwenden, um dominante Farben zu finden
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(all_pixels)
            colors = kmeans.cluster_centers_.astype(int)

            # Originalbeispiele anzeigen
            img_collage = np.hstack(
                [
                    cv2.resize(
                        cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (100, 100)
                    )
                    for p in category_samples[:5]
                    if cv2.imread(p) is not None
                ]
            )

            if len(self.categories) == 1:
                axes[0].imshow(img_collage)
                axes[0].set_title(f"{category} Beispiele")
                axes[0].axis("off")

                # Farbpalette anzeigen
                color_palette = np.ones((100, n_colors * 100, 3), dtype=np.uint8)
                for j, color in enumerate(colors):
                    color_palette[:, j * 100 : (j + 1) * 100] = color

                axes[1].imshow(color_palette)
                axes[1].set_title(f"{category} Dominante Farben")
                axes[1].axis("off")
            else:
                axes[i, 0].imshow(img_collage)
                axes[i, 0].set_title(f"{category} Beispiele")
                axes[i, 0].axis("off")

                # Farbpalette anzeigen
                color_palette = np.ones((100, n_colors * 100, 3), dtype=np.uint8)
                for j, color in enumerate(colors):
                    color_palette[:, j * 100 : (j + 1) * 100] = color

                axes[i, 1].imshow(color_palette)
                axes[i, 1].set_title(f"{category} Dominante Farben")
                axes[i, 1].axis("off")

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