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
    
    # ZMIENIONO: Ładowanie modelu YOLOv8s (small)
    model = YOLO('yolov8s.pt') 
    
    print(f"Rozpoczynanie treningu dla zbioru: {DATA_YAML_PATH}")
    # ZMIENIONO: Komunikat modelu bazowego
    print("Model bazowy: YOLOv8s (small)")
    
    # 3. Rozpoczęcie treningu
    
    results = model.train (
        data=DATA_YAML_PATH,  # Ścieżka do Twojego pliku konfiguracyjnego
        epochs=500,           # Liczba epok (możesz zmienić)
        imgsz=1024,            # Rozmiar wejściowy obrazu
        # UWAGA: Przy YOLOv8s i imgsz=1024, BATCH=24 może być za duży!
        # Rozważ zmniejszenie batcha do 16 lub 12, jeśli wystąpi błąd VRAM.
        batch=16,             
        name='drzwi_projekt_cpov8_s', # Dodaj '_s' do nazwy dla porządku
        workers=0,            # Zmienione na 0 dla Windows
        device=0,
        # 1. OPTYMALIZATOR
        optimizer='AdamW',         # Lepsza stabilność i konwergencja
        lr0=0.001,                 # Mniejsza szybkość uczenia dla precyzji
        # 2. FUNKCJA STRATY
        cls=1.5,                   # Zwiększona waga błędu klasyfikacji (na błąd mylenia klas)
        # 3. WZMOCNIENIE DANYCH
        hsv_v=0.15,                # Zwiększona zmiana jasności/oświetlenia
        translate=0.2,             # Większe losowe przesunięcia
        degrees=5.0                # Lekkie rotacje            
    )
    
    print("\n--- Trening zakończony ---")