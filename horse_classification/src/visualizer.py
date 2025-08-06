import cv2
import numpy as np
import json
from tqdm import tqdm

from .utils import load_config

class Visualizer:
	def __init__(self):
		config = load_config()
		self.skeleton_connections = config['keyjoints_configuration']['skeleton_connections']
		self.output_path = config['paths']['visualizer_output']

		with open(config['paths']['pose_estimation_output'], 'r') as file:
			self.frames_keypoints = json.load(file)
		with open(config['paths']['movement_recognition_output'], 'r') as file:
			self.movement_predictions = json.load(file)
		with open(config['paths']['classification_output'], 'r') as file:
			self.classification = json.load(file)


	def draw_horse_skeleton(self, frame: np.ndarray, keypoints) -> None:
		for i, j in self.skeleton_connections:
			if not ((int(keypoints[i][0]) == 0 and int(keypoints[i][1]) == 0) or (int(keypoints[j][0]) == 0 and int(keypoints[j][1]) == 0)):
				pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
				pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
				cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green lines

		# Draw keypoints
		for x, y in keypoints:
			if not(int(x)== 0 and int(y)==0) :
				cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)


	def run(self, input_video_path: str) -> None:
		cap = cv2.VideoCapture(input_video_path)
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		# Define video writer
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
		out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

		frame_idx = 0
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.8
		font_thickness = 2
		x_pad, y_pad = 10, 10

		with tqdm(total=total_frames, desc="Visualizing", unit="frame") as pbar:
			while cap.isOpened():
				ret, frame = cap.read()
				if not ret:
					break

				# Determine movement label for current frame
				movement_label = None
				for prob, movement, start_frame, end_frame in self.movement_predictions:
					if start_frame <= frame_idx <= end_frame:
						movement_label = movement
						break

				# Prepare text and rectangles
				movement_text = movement_label if movement_label else ""
				(label_w, label_h), _ = cv2.getTextSize(movement_text, font, font_scale, font_thickness)
				label_rect_y1 = y_pad
				label_rect_y2 = y_pad + label_h + 10

				class_text = f"Classified as '{self.classification['label']}'"
				(class_w, class_h), _ = cv2.getTextSize(class_text, font, font_scale, font_thickness)
				class_rect_y1 = label_rect_y2 + 5
				class_rect_y2 = class_rect_y1 + class_h + 10

				# Draw overlays
				overlay = frame.copy()
				cv2.rectangle(overlay, (x_pad, label_rect_y1), (x_pad + label_w + 10, label_rect_y2), (0, 0, 0), -1)
				cv2.rectangle(overlay, (x_pad, class_rect_y1), (x_pad + class_w + 10, class_rect_y2), (0, 0, 0), -1)
				frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

				# Draw text
				cv2.putText(frame, movement_text, (x_pad + 5, label_rect_y2 - 5), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
				cv2.putText(frame, class_text, (x_pad + 5, class_rect_y2 - 5), font, font_scale, (0, 200, 0), font_thickness, cv2.LINE_AA)

				# Draw skeleton and save frame
				self.draw_horse_skeleton(frame, self.frames_keypoints[frame_idx]['keypoints'])
				out.write(frame)

				frame_idx += 1
				pbar.update(1)  # Update progress bar

		cap.release()
		out.release()
		print(f"The output video is successfully saved as {self.output_path}")