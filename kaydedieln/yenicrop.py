
import cv2
import numpy as np

# Giriş görüntü dosyasının adını belirtin
image_path = (r"C:\Users\ilknur_ozen\Documents\GitHub\yolov8-live\asd.jpg")

# Üçgenin köşe noktalarını belirtin
x1, y1 = 365, 189
x2, y2 = 105, 198
x3, y3 = 247, 13
# Görüntüyü okuyun
image = cv2.imread(image_path)

# Üçgenin içindeki pikselleri mask olarak kullanarak belirtilen bölgeyi alın
mask = np.zeros_like(image)
points = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
cv2.fillPoly(mask, [points], (255, 255, 255))
cropped_region = cv2.bitwise_and(image, mask)

# Belirtilen bölgeyi bir dosyaya kaydedin
output_path = "cropped_triangle_region.jpeg"
cv2.imwrite(output_path, cropped_region, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
print(f"Cropped triangle region saved as {output_path}")
