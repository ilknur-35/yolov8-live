import cv2
import time
import os

def capture_and_save_last_frame(camera_url, output_folder):
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print(f"Error: Couldn't open the camera with URL {camera_url}")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    last_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame from the camera.")
            break

        # Kaydetme işlemini saniyede bir kez yap
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_filename = f"{timestamp}.jpg"
        image_path = os.path.join(output_folder, image_filename)

        cv2.imwrite(image_path, frame)
        print(f"Frame saved: {image_path}")

        # En son frame'i güncelle
        last_frame = frame

        # 1 dakika (60 saniye) bekle
        time.sleep(60)

    cap.release()

    # Döngüden çıkınca en son frame'i kullanabilirsiniz
    if last_frame is not None:
        cv2.imshow("Last Frame", last_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_url = "http://admin:20basm07.@10.26.161.35/stw-cgi/video.cgi?msubmenu=snapshot&action=view"
    output_folder = "captured_frames"

    try:
        capture_and_save_last_frame(camera_url, output_folder)
    except KeyboardInterrupt:
        print("Capture stopped by the user.")