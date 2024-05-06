import numpy as np
from PIL import Image

# Görüntüleri yükle
im0 = np.array(Image.open('C:/Users/ilknur_ozen/Documents/GitHub/yolov8-live/1.jpg'))
im1 = np.array(Image.open('C:/Users/ilknur_ozen/Documents/GitHub/yolov8-live/2.jpg'))
im2 = np.array(Image.open('C:/Users/ilknur_ozen/Documents/GitHub/yolov8-live/3.jpg'))
im3 = np.array(Image.open('C:/Users/ilknur_ozen/Documents/GitHub/yolov8-live/4.jpg'))
im4 = np.array(Image.open('C:/Users/ilknur_ozen/Documents/GitHub/yolov8-live/5.jpg'))

# Görüntüleri aynı boyuta getir
min_shape = np.min([im.shape[:2] for im in [im0, im1, im2, im3, im4]], axis=0)
im0 = im0[:min_shape[0], :min_shape[1]]
im1 = im1[:min_shape[0], :min_shape[1]]
im2 = im2[:min_shape[0], :min_shape[1]]
im3 = im3[:min_shape[0], :min_shape[1]]
im4 = im4[:min_shape[0], :min_shape[1]]

# Stack the 5 images into a 4d sequence
sequence = np.stack((im0, im1, im2, im3, im4), axis=3)

# Replace each pixel by mean of the sequence
result = np.median(sequence, axis=3).astype(np.uint8)

# Save to disk
Image.fromarray(result).save('result.png')
