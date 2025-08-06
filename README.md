# Horse Classification Project

## Horse Pose and Segmentation

This project provides tools for processing horse-related videos. It includes two main components:
1. **Horse Pose Estimation**: Detects and tracks horse keyjoints in videos.
2. **Horse Segmentation**: Segments horse regions in videos.

## Repository Structure
```
/horse_pose_estimation  # Pose estimation module
    - main.py           # Main script for pose estimation
    - README.md         # Instructions for using pose estimation

/horse_segmentation     # Segmentation module
    - main.py           # Main script for segmentation
    - README.md         # Instructions for using segmentation

README.md               # General project overview
requirements.txt        # Dependencies
```

## Installation
Ensure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage
Each module contains a `main.py` script that can be executed independently. Refer to their respective `README.md` files for detailed instructions.

### Running Horse Pose Estimation
```bash
python horse_pose_estimation/main.py --input_video_path <input.mp4> --output_video_path <output.mp4>
```

### Running Horse Segmentation
```bash
python horse_segmentation/main.py --input_video_path <input.mp4> --output_video_path <output.mp4>
```
