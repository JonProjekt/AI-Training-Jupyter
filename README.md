# YOLO11n Object Detection Toolkit

Dieses Projekt enthält verschiedene Python-Skripte für die Arbeit mit annotierten Bilddatensätzen, YOLO-Training und die professionelle Auswertung und Visualisierung von Resultaten. Der Schwerpunkt liegt auf industriellen Klassifikations- und Qualitätsaufgaben im Spritzgussumfeld und auf der schnellen, flexiblen Anpassung aller Parameter.

---

## **Inhalt**

### **1. Datenannotation** (nicht enthalten)

- Dieser Schritt wird extern durchgeführt.
- Es wird empfohlen, den [Label Studio](https://labelstud.io/) zu verwenden, um Bilder zu annotieren und die Annotationen im YOLO-Format zu exportieren.

### **2. Augmentation & Data Preparation**
- **augment_images.py**  
  - Lädt einen annotierten Bilddatensatz (YOLO-Format) und vervielfältigt die Daten mittels Rotation, Zoom, Verzerrung, Spiegelung, Farbveränderung, Blurring und Helligkeitsanpassung.
  - Ermöglicht per Schalter das Aktivieren/Deaktivieren einzelner Augmentationen.
  - Split nach Ultralytics-Standard in train/val/test und automatische Erstellung einer dataset.yaml.

### **2. Training & Hyperparameter Tuning**
- **train_yolo.py**  
  - Führt das Training eines YOLOv8 nano Modells (yolo11n.pt) durch.
  - Erlaubt flexible Wahl aller Trainingsargumente (Batchgröße, Epochs, Augmentationen etc.).
  - Nutzt GPU (CUDA), wenn verfügbar.

- **tune_conf_iou.py**  
  - Sucht automatisiert die optimalen Konfidenz- und IoU-Schwellen pro Klasse (und/oder global) durch Heatmap-Gridsearch.
  - Erstellt Visualisierungen und eine CSV-Datei mit den gefundenen Schwellen.

### **3. Validierung & Berichterstellung**
- **validate_yolo.py**  
  - Lädt die besten Schwellen aus der Tuning-CSV und wertet das trainierte Modell auf dem Testdatensatz aus.
  - Erstellt Confusion Matrix, tabellarische Auswertungen und einen vollständigen Bericht (Markdown, PNG, optional PDF).

### **4. Live Galerie / Visualisierung**
- **yolo_gallery.py**  
  - Interaktive Bildergalerie: Annotiert einen beliebigen Ordner (z. B. Test/Train/Val) Bild für Bild mit den YOLO-Predictions.
  - Anzeige der Inferenzzeit pro Bild, Vor-/Zurückblättern mit Pfeiltasten, kein Abspeichern – rein zum Durchklicken und Anschauen.

---

## **Installation**

1. Empfohlene Python-Version: **3.10 oder 3.11**
2. Erzeuge ein frisches virtuelles Environment (empfohlen):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate     # Windows
   ```