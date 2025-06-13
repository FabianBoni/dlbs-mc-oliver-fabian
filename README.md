# YOLO-basierte Hundeerkennung - Optimierte Objektdetektion

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Eine umfassende Studie zur Optimierung von YOLO-Architekturen für die automatisierte Erkennung von Hunden in Bildern mit progressiven Trainingsstrategien und systematischer Datenaugmentation.

## 📋 Inhaltsverzeichnis

- [Projektübersicht](#projektübersicht)
- [Installation](#installation)
- [Datensätze](#datensätze)
- [Experimente](#experimente)
- [Ergebnisse](#ergebnisse)
- [Nutzung](#nutzung)
- [Projektstruktur](#projektstruktur)
- [Autoren](#autoren)

## 🎯 Projektübersicht

### Forschungsfrage

Wie zuverlässig erkennen und lokalisieren gängige Object Detection Modelle Hunde in verschiedenenalltäglichen Szenarien, und welche Einflussfaktoren (z.B. Überlappung, Lichtverhältnisse) führen zusignifikanten Leistungseinbussen?

### Hauptziele

- **Baseline-Entwicklung**: Training verschiedener YOLO-Varianten (YOLOv8n, YOLOv8s, YOLOv8m)
- **Progressive Trainingsoptimierung**: Stufenweises Training mit steigenden Auflösungen
- **Datensatz-Evaluation**: Vergleichende Analyse von vier verschiedenen Hundedatensätzen
- **Datenaugmentation**: Systematische Verbesserung kleinerer Datensätze
- **Fehleranalyse**: Identifikation typischer Schwachstellen und Optimierungspotenziale

### 🔧 Technischer Stack

- **Framework**: Ultralytics YOLO (v8)
- **Programmiersprache**: Python 3.8+
- **ML-Bibliotheken**: PyTorch, TensorFlow
- **Datenverarbeitung**: NumPy, Pandas, OpenCV
- **Visualisierung**: Matplotlib, Seaborn
- **Augmentation**: Albumentations
- **Dokumentation**: Quarto

## 🚀 Installation

### 1. Repository klonen

```bash
git clone https://github.com/username/dlbs-mc-oliver-fabian.git
cd dlbs-mc-oliver-fabian
```

### 2. Virtuelle Umgebung erstellen

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS  
source .venv/bin/activate
```

### 3. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 4. Zusätzliche YOLO-Installation

```bash
pip install ultralytics
```

## 📊 Datensätze

Das Projekt verwendet vier öffentlich verfügbare Datensätze:

| Datensatz                  | Beschreibung                              | Anzahl Bilder |
| -------------------------- | ----------------------------------------- | ------------- |
| **Kaggle Dog & Cat** | Gemischter Datensatz, auf Hunde gefiltert | 3686          |
| **Dogs OVDDC**       | Spezialisierter Hundedatensatz            | 531           |
| **ASDF-T4TSD**       | Kleiner, spezialisierter Datensatz        | 36            |
| **Max EVO5Q Dog**    | Kompakter Hundedatensatz                  | 108           |

### Datenvorbereitung

Alle Datensätze wurden standardisiert:

- Konvertierung zu YOLO-Format
- Einheitliche Klassendefinition ("dog" = Klasse 0)
- 70/15/15 Split (Train/Valid/Test)
- Bereinigung fehlerhafter Annotationen

## 🔬 Experimente

### 1. Baseline-Modelle

Training von drei YOLO-Varianten auf standardisierten Datensätzen:

- **YOLOv8n**: ~3.2M Parameter, optimiert für Geschwindigkeit
- **YOLOv8s**: ~11.2M Parameter, Balance zwischen Geschwindigkeit und Genauigkeit
- **YOLOv8m**: ~25.9M Parameter, fokussiert auf maximale Genauigkeit

### 2. Progressive Trainingsstrategie

Mehrstufiges Training mit steigenden Auflösungen:

- **Stage 1**: 320px Auflösung (Epochen 0-20)
- **Stage 2**: 512px Auflösung (Epochen 21-40)
- **Stage 3**: 640px Auflösung (Epochen 41-60)

### 3. Datenaugmentation

Systematische Verbesserung schwacher Datensätze:

- Horizontale Spiegelung
- Helligkeits-/Kontrastanpassung
- Rotation und Skalierung
- Rauschhinzufügung
- 4-fache Vergrößerung der Trainingsmenge

## 📈 Hauptergebnisse

### 🏆 Beste Performance

- **Progressive Training mit YOLOv8s**: **96.86% mAP@0.5**
- **Quantitative Progression**:
  - Stage 1: 90.06% mAP@0.5
  - Stage 2: 94.33% mAP@0.5 (+4.27%)
  - Stage 3: 96.86% mAP@0.5 (+2.53%)
  - **Gesamtverbesserung**: 6.80%

### 📊 Datensatz-spezifische Ergebnisse

| Datensatz          | Precision    | Recall       | mAP@0.5               | Verbesserung durch Aug. |
| ------------------ | ------------ | ------------ | --------------------- | ----------------------- |
| **kaggle**   | 0.92         | 0.90         | **0.95**        | -                       |
| **dogs_v5i** | 0.89         | 0.84         | 0.88                  | -                       |
| **asdf_v1i** | 0.69 → 0.68 | 0.71         | 0.69 →**0.68** | +139%                   |
| **dog_v1i**  | 0.49 → 0.82 | 0.63 → 0.79 | 0.55 →**0.82** | +49%                    |

### 🔍 Wichtigste Erkenntnisse

1. **YOLOv8s übertraf YOLOv8m** trotz geringerer Komplexität
2. **Datenqualität > Datenmenge**: Saubere Annotationen wichtiger als große Datensätze
3. **Augmentation hochwirksam**: Besonders bei kleinen Datensätzen (+49% bis +139%)
4. **Progressive Training effektiv**: Konsistente Verbesserungen bei allen Modellen

## 💻 Nutzung

### Notebooks ausführen

1. **Baseline-Experimente**:

```bash
cd notebooks
quarto render baseline.qmd
```

2. **Datensatz-spezifische Analyse**:

```bash
quarto render einzelnes_training.qmd
```

3. **Umfassende Analyse**:

```bash
quarto render combined_analysis.qmd
```

### Eigene Modelle trainieren

```python
from ultralytics import YOLO

# Modell laden
model = YOLO('yolov8n.pt')

# Training starten
model.train(
    data='path/to/data.yaml',
    epochs=20,
    imgsz=640,
    batch=8
)

# Validierung
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.3f}")
```

### Vorhersagen erstellen

```python
# Modell für Inferenz laden
model = YOLO('runs/train/weights/best.pt')

# Einzelbild-Vorhersage
results = model('path/to/image.jpg')

# Batch-Vorhersage
results = model(['img1.jpg', 'img2.jpg'])

# Ergebnisse anzeigen
for result in results:
    result.show()
```

## 📁 Projektstruktur

```
dlbs-mc-oliver-fabian/
├── README.md                    # Diese Datei
├── requirements.txt             # Python-Abhängigkeiten
├── .gitignore                  # Git-Ignorierung
│
├── data/                       # Original-Datensätze
│   ├── train/
│   ├── valid/
│   └── test/
│
├── data_small/                 # Kleine Testdatensätze
│   └── dog_dataset.yaml
│
├── notebooks/                  # Jupyter/Quarto Notebooks
│   ├── baseline.qmd           # Baseline-Experimente
│   ├── einzelnes_training.qmd # Datensatz-Analyse
│   ├── combined_analysis.qmd  # Umfassende Analyse
│   └── eda/                   # Explorative Datenanalyse
│
├── baseline/                   # Baseline-Modellgewichte
│   └── best_model.keras
│
├── runs/                      # YOLO Trainingsergebnisse
│   └── detect/
│
├── scripts/                   # Hilfsskripte
│   ├── xml_to_yolo.py        # Format-Konvertierung
│   └── make_small_subset.py  # Datensatz-Verkleinerung
│
└── eda/                       # EDA-Module
    ├── __init__.py
    └── visualisation.py
```

## 🔧 Erweiterte Konfiguration

### GPU-Optimierung

```python
# GPU-Check
import torch
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
print(f"GPU-Anzahl: {torch.cuda.device_count()}")

# YOLO mit GPU
model = YOLO('yolov8n.pt')
model.train(device='0')  # GPU 0 verwenden
```

### Hyperparameter-Tuning

```yaml
# config.yaml Beispiel
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
box: 0.05
cls: 0.5
dfl: 1.5
```

## 📊 Benchmarks

### Geschwindigkeits-Vergleich

| Modell  | Inferenz-Zeit (ms) | FPS | Genauigkeit (mAP@0.5) |
| ------- | ------------------ | --- | --------------------- |
| YOLOv8n | 8.2                | 122 | 74.0%                 |
| YOLOv8s | 12.1               | 83  | **96.9%**       |
| YOLOv8m | 25.3               | 40  | 95.2%                 |

### Hardware-Anforderungen

- **Minimum**: 8GB RAM, GTX 1060 (6GB VRAM)
- **Empfohlen**: 16GB RAM, RTX 3070 (8GB VRAM)
- **Optimal**: 32GB RAM, RTX 4080+ (12GB+ VRAM)

## 🤝 Beitragen

1. Fork des Repositories
2. Feature-Branch erstellen (`git checkout -b feature/AmazingFeature`)
3. Änderungen committen (`git commit -m 'Add AmazingFeature'`)
4. Branch pushen (`git push origin feature/AmazingFeature`)
5. Pull Request öffnen

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe [LICENSE](LICENSE) für Details.

## 👥 Autoren

- **Oliver** - *Baseline-Entwicklung und Progressive Training*
- **Fabian** - *Datensatz-Analyse und Augmentation*

## 🙏 Danksagungen

- [Ultralytics](https://github.com/ultralytics/ultralytics) für das exzellente YOLO-Framework
- Kaggle Community für qualitativ hochwertige Datensätze
- Open-Source Computer Vision Community

## 📧 Kontakt

Bei Fragen oder Anregungen erstellen Sie gerne ein Issue oder kontaktieren Sie uns direkt.

---

**⭐ Wenn Ihnen dieses Projekt gefällt, geben Sie ihm gerne einen Stern!**
