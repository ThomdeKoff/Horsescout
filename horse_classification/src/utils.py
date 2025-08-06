import os
import yaml
import cv2
from tqdm import tqdm

def compute_iou(box1, box2):
	"""Compute IoU (Intersection over Union) between two bounding boxes."""
	x1_inter = max(box1[0], box2[0])
	y1_inter = max(box1[1], box2[1])
	x2_inter = min(box1[2], box2[2])
	y2_inter = min(box1[3], box2[3])

	inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
	box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
	box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

	union_area = box1_area + box2_area - inter_area
	return inter_area / union_area if union_area else 0

def load_config(config_filename="./config/config.yaml"):

	config_path = os.path.abspath(config_filename)
	config_dir = os.path.dirname(config_path)

	# Load YAML config
	with open(config_path, 'r') as file:
		config = yaml.safe_load(file)

	# Convert relative paths to absolute paths
	if 'paths' in config:
		for key, rel_path in config['paths'].items():
			# Join with config directory if path is not already absolute
			if not os.path.isabs(rel_path):
				config['paths'][key] = os.path.normpath(os.path.join(config_dir, rel_path))

	return config



def get_video_fps_height(video_path):
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Cannot open video: {video_path}")
		return None

	fps = cap.get(cv2.CAP_PROP_FPS)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	cap.release()
	return fps, height

def reduce_video_resolution_1080p(input_video_path, output_video_path):

		# Load input video
		cap = cv2.VideoCapture(input_video_path)

		# Get video properties
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = cap.get(cv2.CAP_PROP_FPS)
		width, height = 1920, 1080  # Target 1080p resolution

		# Define video writer
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

		# Process frames with progress bar
		for _ in tqdm(range(frame_count), desc="Resizing video"):
			ret, frame = cap.read()
			if not ret:
				break
			resized = cv2.resize(frame, (width, height))
			out.write(resized)

		# Release resources
		cap.release()
		out.release()

