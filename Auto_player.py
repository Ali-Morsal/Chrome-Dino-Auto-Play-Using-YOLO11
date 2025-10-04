import cv2
import numpy as np
import mss
import pyautogui
from ultralytics import YOLO
import time

def main():
    model = YOLO(r"C:\Users\Ali-M\Desktop\Deep Projects\Dino Game auto play\best.pt") # Adjust according to your path

    """
        These values are for a 1920x1080 screen resolution. 
        When chrome://dino is 150% zoomed and browser is maximized. 
        Adjust if necessary.
    """
    monitor = {

        "top": 186,
        "left": 510,
        "width": 900,
        "height": 224
    }

    # Distance threshold settings
    DISTANCE_THRESHOLD_PERCENT = 0.15111
    # For adjusting the distance threshold over time due to increasing game speed
    ONE_PERCENT_OF_DTP = DISTANCE_THRESHOLD_PERCENT / 100


    image_width = monitor["width"]
    image_height = monitor["height"]

    dino_class_id = 2
    cactus_class_id = 0
    ptero_class_id = 1

    frame_count = 0
    start_time = time.time()
    elapsed_time = 0
    accelerator_counter = 1

    # Cooldown settings inorder to prevent multiple jumps for the same obstacle
    last_jump_time = 0.0
    JUMP_COOLDOWN = 0.2  # seconds

    # Margins to ignore detections too close to the edges
    LOW_MARGIN_PX = max(8, int(image_height * 0.05))

    with mss.mss() as sct:
        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            results = model(frame, verbose=False, imgsz=416)[0]

            dino_x_center = None
            cactus_x_center = None
            ptero_x_center = None
            dino_y_top = None
            dino_y_bottom = None
            ptero_y_down = None
            ptero_y_top = None

            for box in results.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x_center = (x1 + x2) / 2
                y_top = y1
                y_down = y2

                if cls == dino_class_id:
                    dino_x_center = x_center
                    dino_y_top = y1
                    dino_y_bottom = y2
                elif cls == cactus_class_id:
                    if cactus_x_center is None or x_center < cactus_x_center:
                        cactus_x_center = x_center
                elif cls == ptero_class_id:
                    if ptero_x_center is None or x_center < ptero_x_center:
                        ptero_x_center = x_center
                        ptero_y_down = y2
                        ptero_y_top = y1

            # Logics for jumping over cactus
            elapsed_time = time.time() - start_time
            if dino_x_center is not None and cactus_x_center is not None:
                distance_percent = (cactus_x_center - dino_x_center) / image_width
                if 0 < distance_percent <= DISTANCE_THRESHOLD_PERCENT:
                    if time.time() - last_jump_time > JUMP_COOLDOWN:
                        pyautogui.press("up")
                        last_jump_time = time.time()
                        print(f"{elapsed_time:.4f}: Cactus Jump triggered! | Distance (Cactus) = {distance_percent:.4f}, d-top={dino_y_top}, d-down={dino_y_bottom}")

            # Logics for jumping over pteros
            if dino_x_center is not None and ptero_x_center is not None:
                distance_percent_ptero = (ptero_x_center - dino_x_center) / image_width

                if 0 < distance_percent_ptero <= DISTANCE_THRESHOLD_PERCENT:
                    if None in (ptero_y_top, ptero_y_down, dino_y_top, dino_y_bottom):
                        print(f"{elapsed_time:.4f}: Missing Y coords for ptero/dino -> skip")
                        continue

                    # For when ptero is high.        
                    if dino_y_top > ptero_y_top:
                        pyautogui.keyDown("down")
                        time.sleep(0.2)
                        pyautogui.keyUp("down")
                        print(
                            f"{elapsed_time:.4f}: Duck triggered | Distance = {distance_percent_ptero:.4f}, p-top={ptero_y_top}, p-down={ptero_y_down}, d-top={dino_y_top}, d-down={dino_y_bottom}")

                    # For when ptero is low
                    elif ptero_y_top > dino_y_top:
                        pyautogui.press("up")
                        print(
                            f"{elapsed_time:.4f}: Jump triggered | Distance = {distance_percent_ptero:.4f}, p-top={ptero_y_top}, p-down={ptero_y_down}, d-top={dino_y_top}, d-down={dino_y_bottom}")

                    # just for debugging purposes
                    else:
                        print(f"{elapsed_time:.4f}: no action | Distance = {distance_percent_ptero:.4f}, p-top={ptero_y_top}, p-down={ptero_y_down}, d-top={dino_y_top}, d-down={dino_y_bottom}")    

            # increase distance threshhold to match game speed over time
            if elapsed_time > accelerator_counter:  
                accelerator_counter += 1
                DISTANCE_THRESHOLD_PERCENT += ONE_PERCENT_OF_DTP
                print(f"Elapsed time: {elapsed_time:.1f}s | DISTANCE_THRESHOLD_PERCENT={DISTANCE_THRESHOLD_PERCENT:.4f}")

            frame_count += 1
            play_time = 200  # Playtime in seconds
            if frame_count >= play_time * 60: 
                break

if __name__ == "__main__":
    main()
