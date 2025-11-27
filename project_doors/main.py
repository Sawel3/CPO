import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from pytesseract import Output
from datetime import datetime
import os
import math

# KONFIGURACJA
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

LOG_FILE = "wyniki_detekcji.txt"
CLASS_NAMES = {0: 'drzwi', 1: 'nr_sali', 2: 'tabliczka_info', 3: 'apteczka'}

def log_to_file(text_line):
    """Dopisuje linię do pliku tekstowego."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text_line + "\n")

def process_pipeline(image_path, model_path='best.pt'):
    # 1. Wczytanie
    img = cv2.imread(image_path)
    if img is None: return

    # --- Operacje geometryczne (skalowanie do 1024) ---
    target_width = 1024
    scale = target_width / img.shape[1]
    h, w = int(img.shape[0] * scale), target_width
    img_resized = cv2.resize(img, (w, h))

    # --- Filtracja ---
    img_filtered = cv2.GaussianBlur(img_resized, (3, 3), 0)

    # 2. Detekcja YOLO
    model = YOLO(model_path)
    results = model(img_filtered, imgsz=1024, conf=0.4, verbose=False)

    output_img = img_filtered.copy()
    
    # Znacznik czasu do logów
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(f"--- Analiza pliku: {image_path} [{timestamp}] ---")

    for r in results:
        for box in r.boxes:
            # Dane z YOLO
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            yolo_conf = float(box.conf[0])
            class_name = CLASS_NAMES.get(cls_id, 'obj')

            roi = img_filtered[y1:y2, x1:x2]
            if roi.size == 0: continue

            # Domyślne wartości dla obiektów bez tekstu
            final_label = f"{class_name} ({yolo_conf:.2f})"
            ocr_text = "N/A"
            ocr_conf = 0.0

            # === OCR tylko dla numerów sal ===
            if cls_id == 1: 
                # Preprocessing dla OCR (powiększenie + binaryzacja)
                roi_ocr = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(roi_ocr, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Konfiguracja: psm 7 (jedna linia), tylko cyfry i duże litery
                custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                
                # Używamy image_to_data aby dostać pewność (conf)
                data = pytesseract.image_to_data(binary, config=custom_config, output_type=Output.DICT)
                
                # Filtrowanie pustych wyników i szukanie tekstu z najwyższą pewnością
                found_texts = []
                found_confs = []
                
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    if int(data['conf'][i]) > 0 and data['text'][i].strip():
                        found_texts.append(data['text'][i])
                        found_confs.append(int(data['conf'][i]))

                if found_texts:
                    ocr_text = " ".join(found_texts)
                    ocr_conf = sum(found_confs) / len(found_confs) # Średnia pewność
                    final_label = f"SALA: {ocr_text} (OCR:{ocr_conf:.0f}%)"
                else:
                    ocr_text = "[Nieczytelne]"
                    final_label = f"SALA: ?"

            # --- ZAPIS DO PLIKU ---
            # Format: KLASA | YOLO_CONF | OCR_TEXT | OCR_CONF
            log_line = f"Obiekt: {class_name:<15} | YOLO: {yolo_conf:.2%} | Tekst: {ocr_text:<10} | OCR Pewność: {ocr_conf:.0f}%"
            print(log_line)
            log_to_file(log_line)

            # --- Morfologia ---
            kernel = np.ones((3,3), np.uint8)
            # Operacja zamknięcia
            _ = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

            # Rysowanie
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_img, final_label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- ZAPIS OBRAZU WYJŚCIOWEGO ---
    # Domyślnie zapisuje obok pliku wejściowego z sufiksem "_output"
    save_path = os.path.splitext(image_path)[0] + "_output.jpg"
    cv2.imwrite(save_path, output_img)
    log_to_file(f"Zapisano obraz wyjściowy: {save_path}")

    cv2.imshow("System Detekcji", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_pipeline('20251106_171353.jpg', 'best2.pt')
