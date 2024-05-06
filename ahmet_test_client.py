import io
from ahmet_test_class_file import ImageProcessor
  
import os
import cv2
import numpy as numpy
from PIL import Image


# test variables
confidence = 0.2
#image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\x00\x00\x00\x00IEND\xaeB`\x82'

# read image from file  "ds.jpg"
image_path = "ds.jpg"

# need bytes array, itll bve converted like this: image = Image.open(io.BytesIO(image_bytes))
image_bytes = open(os.path.join(os.getcwd(), image_path), "rb").read()

                   

rule_polygon_list_1 = numpy.array(
    [[697,153], [542,205], [375,284], 
    [777,728],[1044,547]
    ])

rule_polygon_list_2 = numpy.array(
    [
    [1436,1060], [1015,1060], [958,1235], [962,1358], [1057,1441],
    [1283,1431]
    ])


# 1 liste var diyelim,

point_array = [rule_polygon_list_1,]

# 2 veya daha fazla liste var diyelim,
point_array = [rule_polygon_list_1, rule_polygon_list_2,]

deniz_cop_model = "best (3).pt"

# ImageProcessor sınıfını oluşturur
deniz_cop_tespit_modulu = ImageProcessor(model_directory=deniz_cop_model)

#
processed_image, detection = deniz_cop_tespit_modulu.detect(image_bytes, point_array, confidence)

# create Image from bytes to show
image = Image.open(io.BytesIO(processed_image))
image = numpy.array(image)

# show the processed image
cv2.imshow("Trash Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(detection)

# wait for terminal input
input("Tekrar çalıştırmak için bir tuşa basın...")

while True:
    #
    processed_image, detection = deniz_cop_tespit_modulu.detect(image_bytes, point_array, confidence)

    # create Image from bytes to show
    image = Image.open(io.BytesIO(processed_image))
    image = numpy.array(image)

    # show the processed image
    cv2.imshow("Trash Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    continue