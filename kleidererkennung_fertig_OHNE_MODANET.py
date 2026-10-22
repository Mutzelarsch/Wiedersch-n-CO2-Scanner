import ultralytics
from ultralytics import YOLO
# from ultralytics import Modanet (Modanet als 2. Modell für Mützen usw.)
import cv2


# 1. Modell laden
try:
    # Falls die Datei in einem Unterordner liegt, Pfad anpassen
    model = YOLO("deepfashion2_yolov8s-seg.pt")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    exit()

# 2. CO2-Werte Mapping
# WICHTIG: Die Keys hier sind klein geschrieben und nutzen Leerzeichen statt Unterstriche
co2_data = {
    "short sleeved shirt": 5.2,
    "long sleeved shirt": 8.1,
    "short sleeve outerwear": 12.5,
    "long sleeve outerwear": 18.3,
    "vest": 6.4,
    "sling": 3.1,
    "shorts": 7.8,
    "trousers": 15.0,
    "skirt": 9.5,
    "short sleeve dress": 14.2,
    "long sleeve dress": 19.8,
    "vest dress": 12.7,
    "sling dress": 11.4
}

# 3. Webcam initialisieren
cap = cv2.VideoCapture(1)  # Probiere 0 oder 1

if not cap.isOpened():
    print("Error: Kamera konnte nicht geöffnet werden.")
    exit()

print("Programm gestartet. Drücke 'q' zum Beenden.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferenz (stream=True spart Speicher bei Videos)
    results = model(frame, stream=True)

    for r in results:
        # Wir loopen durch jede erkannte Box
        for box in r.boxes:
            # Koordinaten extrahieren
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Klassennamen vom Modell holen
            cls_id = int(box.cls[0])
            model_label = model.names[cls_id]

            # NORMALISIERUNG (Fix für das 0.0kg Problem)
            # Macht 'Short_Sleeve_Top' zu 'short sleeve top'
            clean_name = model_label.lower().strip().replace("_", " ")

            # Wert aus Dictionary abrufen
            savings = co2_data.get(clean_name, 0.0)

            # Debug-Ausgabe in der Konsole, falls ein Wert fehlt
            if savings == 0.0:
                print(f"Hinweis: '{clean_name}' hat keinen CO2-Wert im Dictionary.")

            # --- Zeichnen ---
            # Farbe für die Box (BGR: Grün)
            color = (0, 255, 0)

            # Text zusammenbauen
            display_text = f"{clean_name.upper()} | {savings}kg CO2"

            # Box zeichnen
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Hintergrund-Balken für Text (für bessere Lesbarkeit)
            (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), color, -1)

            # Text auf das Bild schreiben (Weiß auf grünem Grund)
            cv2.putText(frame, display_text, (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 4. Fenster anzeigen
    cv2.imshow("DeepFashion2 Eco-Scanner", frame)

    # Beenden mit Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()