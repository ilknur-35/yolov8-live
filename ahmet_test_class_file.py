import os
import cv2
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas
import json

class ImageProcessor:
    def __init__(self, model_directory: str, target_size=(1600, 900), img_sz=800):
        self.model_directory = model_directory
        self.target_size = target_size
        self.img_sz = img_sz
        self.detection_model = YOLO(os.path.join(os.getcwd(), self.model_directory))
        
#detection = deniz_cop_tespit_modulu.detect(image_bytes, point_array, confidence)

    def detect(self, image_bytes: bytes, point_arrays_list: list[np.ndarray], confidence: float):
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_bytes))
            image = np.array(image)
            
            selected_polygons_images = []
            
            # Mask the image with the given points array, invert the mask and apply it to the image
            for point_array in point_arrays_list:
                mask = np.ones(image.shape[:2], dtype="uint8") * 255
                cv2.fillPoly(mask, [point_array], 0)
                mask_inv = cv2.bitwise_not(mask)
                selected_polygons_image = cv2.bitwise_and(image, image, mask=mask_inv)
                selected_polygons_images.append(selected_polygons_image)
            
            # now merge the selected polygons images
            merged_image = np.zeros_like(image)
            for selected_polygons_image in selected_polygons_images:
                merged_image = cv2.add(merged_image, selected_polygons_image)
                
            
            # show merged image
            cv2.imshow("Merged Image", merged_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # make image grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # make grayscale image 3 channel
            grayscale_base_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            
            
            # Detect trash
            results = self.detection_model.predict(merged_image, imgsz=self.img_sz, conf=confidence, save=False)
            labeled_image = results[0].plot()
            
            # overlay cropped image on grayscale image
            labeled_image = cv2.addWeighted(labeled_image, 1, grayscale_base_image, 1, 0)
            
            # convert labeled image to bytes
            _, buffer = cv2.imencode(".jpg", labeled_image)
            labeled_image_bytes = buffer.tobytes()
            
            
            # we need detections like this: {"x": 0, "y": 0, "width": 100, "height": 100, "confidence": 0.5, "label": "trash", "label_id": 0}, ...
            detections = []
            
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls.item())
                    xyxy = box.xyxy.tolist()[0]
                    confidence = box.conf.item()
                    x_start, y_start, x_end, y_end = map(int, xyxy)
                    
                    detections.append({"x_start": x_start, "y_start": y_start, "x_end": x_end, "y_end": y_end, "confidence": confidence, "label": cls})
                    
                    
            return labeled_image_bytes, detections
        
        except Exception as e:
            print(e)
            return []