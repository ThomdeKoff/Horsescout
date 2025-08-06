# 🐎 Horse Classification Pipeline

This project provides a complete pipeline to analyze horse videos, detect pose keypoints using YOLO, recognize movements like **jumping** and **running**, analyze these instances and finally classify the horse as **good** or **bad** based on a machine learning model.

---

## 🚀 How to Run

### 1. Install Dependencies

Make sure you have **Python 3.8+** installed. Then, install all required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Main Script

```bash
python main.py --input_video_path path/to/your/horse_video.mp4
```
To disable annotated video output:
```bash
python main.py --input_video_path path/to/video.mp4 --no-visualize
```
## Example Output

```bash
Video processing: pose estimation...
Processing Video: 100% 5091/5091 [01:55<00:00, 43.94frame/s]
Horse movements recognition...
Horse data analysis...
The horse is classified as 'bad horse'
Building the annotated video...
Visualizing: 100% 5091/5091 [01:56<00:00, 43.63frame/s]
The output video is successfully saved as /horse_classification/data/output_video.mp4
```
