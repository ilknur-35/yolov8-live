import cv2
import os

# Klasör yolunu belirtin
folder_path = "C:/Users/ilknur_ozen/Desktop/copResimleri"

# XYWH değerlerini belirtin
x, y, w, h = 1500, 400, 1850, 650

# Klasördeki tüm dosya adlarını alın
file_names = os.listdir(folder_path)

# Her bir dosya için işlemi gerçekleştirin
for file_name in file_names:
    # Dosya yolunu oluşturun
    file_path = os.path.join(folder_path, file_name)

    # Dosyayı okuyun
    image = cv2.imread(file_path)

    # Belirtilen bölgeyi alın
    cropped_region = image[y:y+h, x:x+w]

    # Dosya adını ve uzantısını ayırın
    base_name, extension = os.path.splitext(file_name)

    # Kırpılmış görüntüyü bir dosyaya kaydet
    output_file_path = os.path.join(folder_path, f"cropped_{base_name}.jpeg")
    cv2.imwrite(output_file_path, cropped_region, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print(f"Cropped region saved as {output_file_path}")

print("Cropping completed.")