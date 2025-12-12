import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from pytesseract import Output
from datetime import datetime
import os
import math

# === KONFIGURACJA ŚRODOWISKA I STAŁE ===
# Ścieżka do silnika Tesseract (wymagane w środowisku Windows).
# Bez tego wywołania biblioteka nie znajdzie pliku wykonywalnego.
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

LOG_FILE = "wyniki_detekcji.txt"

# Mapowanie klas: YOLO zwraca numeryczne ID klas (0, 1...). 
# Musimy je zmapować na czytelne etykiety tekstowe dla użytkownika końcowego.
CLASS_NAMES = {0: 'drzwi', 1: 'nr_sali', 2: 'tabliczka_info', 3: 'apteczka'}

def log_to_file(text_line):
    """
    Moduł logowania: Zapisuje wyniki detekcji do pliku tekstowego w trybie 'append'.
    Umożliwia archiwizację wyników i późniejszą analizę statystyczną bez GUI.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text_line + "\n")

def process_pipeline(image_path, model_path='best.pt'):
    # 1. Akwizycja obrazu
    img = cv2.imread(image_path)
    
    # Obsługa błędu odczytu (np. brak pliku, uszkodzony format).
    # Krytyczne dla stabilności pipeline'u przy przetwarzaniu.
    if img is None: 
        print(f"Błąd: Nie można wczytać pliku {image_path}")
        return

    # --- Normalizacja histogramu ---
    # Zastosowano normalizację MinMax do zakresu [0, 255].
    # Uzasadnienie: Zdjęcia z korytarza często mają niską dynamikę tonalną. 
    # Rozciągnięcie histogramu zwiększa kontrast globalny, co ułatwia ekstrakcję cech przez warstwy konwolucyjne YOLO.
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # --- Skalowanie obrazu ---
    # Standaryzacja wejścia: Przeskalowanie do stałej szerokości 1024px z zachowaniem proporcji (aspect ratio).
    target_width = 1024
    scale = target_width / img.shape[1]
    h, w = int(img.shape[0] * scale), target_width
    img_resized = cv2.resize(img, (w, h))

    # --- Filtracja dolnoprzepustowa (Gauss) ---
    # Decyzja projektowa: Zastosowano kernel (1, 1), co de facto wyłącza filtrowanie (bypass).
    # Uzasadnienie: Standardowy kernel (3, 3) działał jako filtr dolnoprzepustowy, który zbyt mocno wygładzał krawędzie
    # na małych obiektach. Powodowało to spadek precyzji detekcji. Funkcja pozostawiona w pipeline, aby w przyszłości
    # łatwo włączyć odszumianie dla sensorów o wysokim ISO.
    img_filtered = cv2.GaussianBlur(img_resized, (1, 1), 0)

    # 2. Inferencja (Detekcja YOLO)
    # Inicjalizacja modelu i wykonanie forward pass.
    model = YOLO(model_path)
    # imgsz=1024: Dopasowanie tensora wejściowego do rozdzielczości obrazu.
    # conf=0.4: Punkt pracy dobrany eksperymentalnie (balans między Precision a Recall).
    results = model(img_filtered, imgsz=1024, conf=0.4, verbose=False)

    # Kopia obrazu do wizualizacji (na niej będziemy rysować, oryginał zostaje czysty do obliczeń).
    output_img = img_filtered.copy()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(f"--- Analiza pliku: {image_path} [{timestamp}] ---")

    # === Post-processing wyników ===
    for r in results:
        for box in r.boxes:
            # Dekodowanie tensora wyjściowego: współrzędne ramki, klasa, pewność.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            yolo_conf = float(box.conf[0])
            class_name = CLASS_NAMES.get(cls_id, 'obj')

            # Wycięcie Region of Interest (ROI) do dalszej analizy.
            roi = img_filtered[y1:y2, x1:x2]
            
            # Zabezpieczenie przed błędami brzegowymi (np. obiekt o powierzchni 0 px).
            if roi.size == 0: continue

            final_label = f"{class_name} ({yolo_conf:.2f})"
            ocr_text = "N/A"
            ocr_conf = 0.0

            # === Pipeline OCR (Tylko dla klasy 'nr_sali') ===
            if cls_id == 1: 
                # Upsampling ROI (interpolacja sześcienna):
                # Zwiększenie rozdzielczości przestrzennej 3-krotnie, aby dostarczyć silnikowi OCR więcej pikseli na znak.
                roi_ocr = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                
                # --- Konwersja przestrzeni barw BGR -> HSV ---
                # Ekstrakcja kanału V (Value/Luminancja).
                # Uzasadnienie: Uniezależnienie algorytmu od chrominancji (koloru oświetlenia). 
                hsv = cv2.cvtColor(roi_ocr, cv2.COLOR_BGR2HSV)
                v_channel = hsv[:, :, 2]
                
                # --- Segmentacja adaptacyjna (Metoda Otsu) ---
                # Minimalizacja wariancję wewnątrzklasowej dla histogramu bimodalnego.
                _, binary = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # --- Operacje morfologiczne ---
                kernel = np.ones((3,3), np.uint8)
                
                # 1. Otwarcie (Opening): Usuwanie szumu tła.
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # 2. Zamknięcie (Closing): Łączenie rozłącznych fragmentów znaków.
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # Zapisz obraz "debug"
                debug_filename = f"debug_roi_{x1}_{y1}.jpg"
                cv2.imwrite(debug_filename, binary)
                print(f"--> Zapisano obraz przetwarzania: {debug_filename}")

                # Konfiguracja Tesseract
                # --psm 7: Page Segmentation Mode = Single Text Line.
                custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                
                data = pytesseract.image_to_data(binary, config=custom_config, output_type=Output.DICT)
                
                found_texts = []
                found_confs = []
                
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    # Filtracja wyników o zerowej pewności (szum interpretowany jako tekst)
                    if int(data['conf'][i]) > 0 and data['text'][i].strip():
                        found_texts.append(data['text'][i])
                        found_confs.append(int(data['conf'][i]))

                if found_texts:
                    ocr_text = " ".join(found_texts)
                    # Średnia ważona pewności OCR dla całego napisu.
                    ocr_conf = sum(found_confs) / len(found_confs)
                    final_label = f"SALA: {ocr_text} (OCR:{ocr_conf:.0f}%)"
                else:
                    ocr_text = "[Nieczytelne]"
                    final_label = f"SALA: ?"

            # --- Logowanie i Wizualizacja ---
            log_line = f"Obiekt: {class_name:<15} | YOLO: {yolo_conf:.2%} | Tekst: {ocr_text:<10} | OCR Pewność: {ocr_conf:.0f}%"
            print(log_line)
            log_to_file(log_line)

            # Rysowanie bounding boxów i etykiet na obrazie wynikowym.
            # Kolor (0, 255, 0) - zielony w przestrzeni BGR.
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_img, final_label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- ZAPIS I PREZENTACJA WYNIKÓW ---
    # Tworzenie struktury katalogów (jeśli nie istnieje) dla uporządkowania wyjścia.
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True) 

    # Generowanie nazwy pliku wyjściowego z sufiksem, aby nie nadpisać oryginału.
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_output.jpg"
    save_path = os.path.join(output_dir, new_filename)

    # Zapis przetworzonego obrazu na dysk.
    cv2.imwrite(save_path, output_img)
    log_to_file(f"Zapisano obraz wyjściowy: {save_path}")

    # Wyświetlenie okna z wynikiem (wymaga interfejsu graficznego).
    cv2.imshow("System Detekcji", output_img)
    # waitKey(0) wstrzymuje program do naciśnięcia klawisza - umożliwia inspekcję wzrokową.
    cv2.waitKey(0)
    # Sprzątanie zasobów okienkowych OpenCV.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_pipeline('IMG_7404.jpg', 'best2.pt')
