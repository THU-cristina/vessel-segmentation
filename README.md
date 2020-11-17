# vessel-segmentation

Mithilfe dieses Projektes können Blutgefäße auf Netzhautbildern selektiert werden.
Der Schwerpunkt liegt hier auf Netzhautbildern von Frühgeborenen.

Folgende Handware wurde verwendet: Intel i7-7700k, 16GB RAM, GTX 1080Ti, Windows 10
Folgende Packages sind zu installieren:
- python 3.8 (46bit)
- openCV 4.4.0
- keras 2.4.3
- tensorflow 2.3 (enthält unteranderem numpy, scipy, hdf5, ...)
- tensorflow-gpu
- PILLOW 7.2.0
- mathplotlib
- CUDA toolkit 10.1

Zusätzlich müssen:
- CUDA als PATH-Systemvariable eingefügen
- CUDNN Dateien kopieren und in CUDA Installationsdateien einfügen

Hier werden die einzelnen Skripte auf ihre Funktionalität beschrieben:
config.py:
- Pfade zu Skripten und Bilddaten
- alle Konfigurationsvariablen für das prepare_dataset_to_hdf5.py, training.py und pretiction.py


extract_patches.py:
- Hier werden die Patches (48x48) für das Training und Testing erstellt, überlappt, ...
- Patches für das Training: 9500 pro Bild, zufällige Positionen, auch außerhalb des FOV
- Patches für die Prediction (Testing): "Fenster"-Patches mit Schrittweite 5 in Höhe und Breite über das ganze Bild (Schrittweite in config.py festgelegt)


prepare_datasets_to_hdf5.py 
- Erstellt aus den Datensätzen die hd5f Dateien
- Die Blder sind im Unterordner datasets/raw_images


training.py
- Führt das Training aus


prediction.py
- Führt das Testing aus


pre_processing.py
- Hier werden die Datensätze bearbeitet (Graustufenbild, Kontrast, ...)


help_functions.py
- Dieses Skript enthält Funktionsdefinitionen für Bildverarbeitungsprozesse 


generate_mask_and_img.py
- Datensätze werden in die benötigte Dateiformate umgewandelt und abgespeichert (IMAGES)
- Masken werden über Kreiserkennung erstellt


functions.py
- Enthält wie helfp_functions ebenso Funktionen zur Bildverarbeitung (Daten werden aber anders abgespeichert)


image_difference.py
- Wurde in der Arbeit nicht beschrieben
- Mit diesem Skript können zwei Binärbilder geladen werden
- dessen Unterschiede werden bei der Ausgabe farblich markiert
- so kann visuell überprüft werden, welche Vorgehensweise bei der Gefäßsegmentierung besser verläuft


Im Ordner:
- failed automatic segmentation ist das Sktipt zur automatischen Segmentierung für den Trainingsdatensatz vorzufinden. Dieser Ansatz wurde durch die manuelle Segmentierung ersetzt
- dataset sind die Datensätze für das Training und Testing (prediction) vorzufinden
- IMAGES sind alle Bilder vorzufinden (eine etwas genauere Beschreibung ist im Ordner selber)
- results sind alle Ergebnisse vorzufinden


Im Ordner dataset fehlen die HDF5-Dateien. Diese waren für GitHub zu groß. Um diese zu generieren muss das Skript prepare_datasets_to_hdf5.py ausgeführt werden.


Das GitHub Repo "https://github.com/orobix/retina-unet" der Firma Orobix S.r.l. wurde als Basis für diese Arbeit genommen.
