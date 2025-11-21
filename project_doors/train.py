import os
from ultralytics import YOLO
# import torch # Opcjonalnie, jeśli chcesz sprawdzić, czy CUDA działa

if __name__ == '__main__':
    
    # 1. Definicja stałych i ścieżek
    
    # Bezwzględna ścieżka do Twojego głównego katalogu projektu
    PROJECT_DIR = 'F:/NoweProgramowanie/Studia/CPO/project_doors'
    
    # Pełna ścieżka do pliku konfiguracyjnego dataset.yaml
    DATA_YAML_PATH = os.path.join(PROJECT_DIR, 'dataset.yaml')
    
    # Sprawdzenie, czy plik istnieje
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Błąd: Nie znaleziono pliku konfiguracyjnego w ścieżce: {DATA_YAML_PATH}")
        print("Upewnij się, że plik dataset.yaml został utworzony!")
        exit()
    
    # 2. Załadowanie modelu
    
    model = YOLO('yolov8n.pt') 
    
    print(f"Rozpoczynanie treningu dla zbioru: {DATA_YAML_PATH}")
    print("Model bazowy: YOLOv8n (nano)")
    
    # 3. Rozpoczęcie treningu
    
    results = model.train(
        data=DATA_YAML_PATH,  # Ścieżka do Twojego pliku konfiguracyjnego
        epochs=500,           # Liczba epok (możesz zmienić)
        imgsz=1024,            # Rozmiar wejściowy obrazu (standard)
        batch=24,             # Rozmiar batcha (dostosuj do pamięci GPU)
        name='drzwi_projekt_cpov8', # Nazwa katalogu wyjściowego w 'runs/detect'
        workers=0,            # Zmień z 4 na 0, aby tymczasowo wyłączyć multiprocessing!
        device=0            # Odkomentuj, jeśli masz GPU
    )
    
    print("\n--- Trening zakończony ---")