import os
import cv2
import numpy as np
from PIL import Image

# Klasör yolunu belirtin
folder_path = "C://Users//ilknur_ozen//Desktop//copResimleri"

# XYWH değerlerini belirtin
x, y, w, h = 1500, 400, 1850, 650

# Klasördeki tüm dosya adlarını alın
file_names = os.listdir(folder_path)

# Her bir dosya için işlemi gerçekleştirin
cropped_images = []
for file_name in file_names:
    # Dosya yolunu oluşturun
    file_path = os.path.join(folder_path, file_name)

    # Sadece JPEG dosyalarını işleyin
    if file_name.lower().endswith(('.jpg', '.jpeg')):
        # Dosyayı okuyun
        image = cv2.imread(file_path)

        # Belirtilen bölgeyi alın
        cropped_region = image[y:y+h, x:x+w]

        # Kırpılmış görüntüyü listeye ekleyin
        cropped_images.append(cropped_region)

# Kırpılmış görüntüleri birleştirin
sequence = np.stack(cropped_images, axis=3)

# Ortanca değeri alın
result = np.median(sequence, axis=3).astype(np.uint8)

# Save to disk
Image.fromarray(result).save('result.png')

# 'result.png' dosyasını açın ve görüntüleyin
with Image.open('result.png') as img:
    img.show()

print("Cropping completed.")