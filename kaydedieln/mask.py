import os
import cv2
import numpy as np
from PIL import Image

# Klasör yolunu belirtin
folder_path = (r"C:\Users\ilknur_ozen\Documents\GitHub\yolov8-live\deneme7")

# Tüm dosya adlarını alın
file_names = os.listdir(folder_path)

# Her bir görüntüyü aynı boyuta getirmek için bir boyut belirleyin
target_size = (1600, 900)  # İhtiyacınıza göre ayarlayın

# Stack için bir liste oluşturun
image_stack = []

# Her bir dosya için işlemi gerçekleştirin
for file_name in file_names:
    # Dosya yolunu oluşturun
    file_path = os.path.join(folder_path, file_name)

    # Dosyayı okuyun1014
    image = cv2.imread(file_path)

    # Görüntüyü belirlenen boyuta yeniden boyutlandırın
    resized_image = cv2.resize(image, target_size)

    # Stack listesine ekleyin
    image_stack.append(resized_image)

# Stack the 3 images into a 4d sequence
sequence = np.stack(image_stack, axis=3)

# Replace each pixel by mean of the sequence
result = np.median(sequence, axis=3).astype(np.uint8)

# Save to disk
# Image.fromarray(result).save('result.png')
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
Image.fromarray(result_rgb).save('result.png')


# 'result.png' dosyasını açın ve görüntüleyin
with Image.open('result.png') as img:
    img.show()
