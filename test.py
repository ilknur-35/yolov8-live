import cv2
import os
from ultralytics import YOLO

def detect_trash(image_path, model):
    # Görüntüyü oku
    image = cv2.imread(image_path)
    
    # Görüntüyü model için uygun formata dönüştür
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    
    # Çöp tespiti yap
    results = model.predict(image)
    
    # Çöpleri kırmızı dikdörtgenlerle işaretle
    for result in results:
        trash_boxes = result.boxes.xyxy
        for box in trash_boxes:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Kırmızı renk, kalınlık: 2
        
    # İşaretlenmiş görüntüyü göster
    cv2.imshow("Trash Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Modeli yükle
    model = YOLO("C:/Users/ilknur_ozen/Documents/GitHub/yolov8-live/.venv/best (3).pt")
    
    # İşlenecek tek bir görüntü dosyası yolu
    image_path = "C:/Users/ilknur_ozen/Documents/GitHub/yolov8-live/captured_frames/20240307_092455.jpg"
    
    # Çöp tespiti yap
    detect_trash(image_path, model)

if __name__ == "__main__":
    main()
