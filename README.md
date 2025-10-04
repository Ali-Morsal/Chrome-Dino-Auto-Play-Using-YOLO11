# Dino Game Auto-Player using YOLO11

This project implements an AI-powered auto-player for the classic Chrome Dino game (accessible at `chrome://dino`). It uses the YOLO11n object detection model to detect the dinosaur, caktus, and pterodactyls in real-time from screen captures. Based on detections, the script automates jumping (over cacti or low-flying pterodactyls) and ducking (under high-flying pterodactyls). The game speed is adaptively handled by increasing the detection threshold over time.

The project includes tools to generate a synthetic dataset, train the YOLO model, and run the auto-player.

## Features
- **Synthetic Dataset Generation**: Creates labeled images mimicking the Dino game using real-label statistics for realistic training data.
- **YOLO Model Training**: Trains a YOLO11n model on the generated dataset to detect classes: Cactus (0), Pterodactyl (1), and Dino (2).
- **Real-Time Gameplay**: Captures the screen, detects obstacles relative to the dino, and sends keyboard inputs (jump with "up", duck with "down").
- **Adaptive Threshold**: Adjusts obstacle detection distance as the game speeds up.
- **Cooldown Mechanism**: Prevents multiple jumps for the same obstacle.
- **Visualization**: Training notebook includes plots for precision-recall curves, confusion matrices, etc.

## Prerequisites
- Python 3.10+ (tested on 3.11).
- A screen resolution of 1920x1080 (adjust monitor coordinates in `Auto_player.py` if different).
- Chrome browser with the Dino game (zoom to 150% for default settings).
- Assets: 
  - Folders: `assets/Bird`, `assets/Cactus`, `assets/Dino` containing PNG/JPG images of pterodactyls, cacti, and dinos (with transparency for overlays).
  - Template images: `assets/day_template.jpg` and `assets/night_template.jpg` (640x160 pixels).
  - `assets/summary.pkl`: Pre-computed label statistics (generate or use provided if available).

## Installation
1. Clone or download the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   This installs:
   - opencv-python==4.11.0.86
   - numpy==1.26.4
   - mss==10.1.0
   - pyautogui==0.9.54
   - ultralytics==8.3.204
   - Pillow==11.3.0
   - PyYAML==6.0.2
   - matplotlib==3.10.5

   Note: For GPU acceleration during training, ensure CUDA is installed and compatible with PyTorch (used by Ultralytics).

## Usage

### 1. Generate Synthetic Dataset
Run `Dataset_Creator.py` to create a YOLO-compatible dataset in the `final_dataset` folder. It splits data into train/valid/test and generates `data.yaml`.

Example command (generates 100 images with 75% train, 15% valid, 10% test):
```
python Dataset_Creator.py 100 0.75 0.15 0.10
```

- Outputs: Images and labels in `final_dataset/{train,valid,test}/{images,labels}`.
- `data.yaml` is auto-generated in `final_dataset/`.
- Customize parameters like image counts or ranges in the script (e.g., `CACTUS_PER_IMAGE_RANGE`).

Ensure assets are in place; the script loads stats from `assets/summary.pkl`.

### 2. Train the YOLO Model
Use `Dino_Detector_Train.ipynb` (preferably in Jupyter Notebook or Kaggle/Colab for GPU support).

- Load the dataset using the path to `final_dataset/data.yaml`.
- Train YOLO11 (nano model by default) for 100 epochs (I used 200 epochs but that was unnecessary).
- Example code in the notebook:
  ```python
  from ultralytics import YOLO
  model = YOLO("yolo11n.pt")  # Start with pre-trained nano model
  yaml_path = "final_dataset/data.yaml"
  model.train(data=yaml_path, epochs=100, imgsz=640, batch=128, device=[0,1], patience=10, workers=2, augment=True)  # Use Double GPU if available
  ```
- After training, evaluate on test set:
  ```python
  results = model.val(data=yaml_path, split="test", save=True)
  ```
- Outputs: Trained model (`best.pt`) in `runs/detect/train/weights/`, along with metrics plots.

Note: Training on a GPU (e.g., NVIDIA Tesla T4 on Kaggle) is recommended for speed.

### 3. Run the Auto-Player
- Update the model path in `Auto_player.py` to your trained `best.pt` or the one that already available.
- Adjust the `monitor` coordinates if your screen setup differs (use tools like `mss` to capture and test regions).
- Start the Chrome Dino game (`chrome://dino`), ensure it's visible and focused.
- Run the script:
  ```
  python Auto_player.py
  ```
- Its recommended to run the script on a CMD which opened as administrator.
- The script captures a screen region, detects objects, and automates controls.
- It runs for ~200 seconds (adjust `play_time` if needed).
- Logs jumps, ducks, and threshold adjustments to console.

**Safety Note**: This uses `pyautogui` for keyboard inputs—ensure no other apps are focused, as it may interfere.

## Project Structure
- `Auto_player.py`: Main script for running the auto-player.
- `Dataset_Creator.py`: Generates synthetic dataset and `data.yaml`.
- `Dino_Detector_Train.ipynb`: Jupyter notebook for training and evaluating the YOLO model.
- `requirements.txt`: List of Python dependencies.
- `assets/`: Folder for game assets (Bird, Cactus, Dino images; templates; summary.pkl).
- `final_dataset/`: Generated dataset (after running Dataset_Creator.py).
- `runs/`: Ultralytics output folder for training results.

## Customization
- **Screen Capture**: Modify `monitor` in `Auto_player.py` for different resolutions or window positions.
- **Thresholds**: Adjust `DISTANCE_THRESHOLD_PERCENT`, `JUMP_COOLDOWN`, or acceleration logic for better performance.
- **Classes**: Cactus (0), Pterodactyl (1), Dino (2)—consistent across dataset and model.
- **Dataset Size**: Increase `num_samples` in Dataset_Creator for better model accuracy (the `best.pt` in the repository trained on a 20000 image dataset).
- **Training Hyperparams**: In the notebook, tweak epochs, batch size, or use a larger YOLO variant (e.g., yolo11s.pt).

## Limitations
- Tested on 1920x1080 resolution; may require tweaks for other setups.
- Performance depends on hardware; real-time detection may lag on low-end CPUs.
- No handling for game over/restart—script runs for a fixed time.

## Contributing
Feel free to fork and submit pull requests for improvements, such as better obstacle handling or multi-resolution support.

## License
This project is licensed under the MIT License—feel free to use and modify.

## Acknowledgments
- Built with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).

If you encounter issues, check console logs or adjust parameters. Happy gaming! 🦖
