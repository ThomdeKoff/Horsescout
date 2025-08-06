import cv2
import numpy as np
from tqdm import tqdm
import os
import logging
from pathlib import Path
from ultralytics import YOLO
import json

from .utils import load_config
from .utils import compute_iou

class HorsePoseEstimation:
	def __init__(self):
		logging.getLogger("ultralytics").setLevel(logging.WARNING)
		config = load_config()
		model_path = config['paths']['pose_estimation_model']
		self.model= YOLO(model_path, verbose = False)
		self.KEYPOINTS_ORDER = config['keyjoints_configuration']['keyjoints_order']
		self.SKELETON_CONNECTIONS = config['keyjoints_configuration']['skeleton_connections']
		self.output_file_path = config['paths']['pose_estimation_output']


	def draw_horse_skeleton(self, frame, keypoints):
		for i, j in self.SKELETON_CONNECTIONS:
			if not ((int(keypoints[i, 0]) == 0 and int(keypoints[i, 1]) == 0) or (int(keypoints[j, 0]) == 0 and int(keypoints[j, 1]) == 0)):
				pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
				pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
				cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green lines

		# Draw keypoints
		for x, y in keypoints:
			cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Draw skeleton lines


	def process_video(self, input_video_path):
		"""
        Processes the video frame by frame, detects keypoints, and visualizes the skeleton.

        Parameters:
        - input_video_path: Path to input video file
        - output_video_path: Path to save output video
        """
		input_video_path = str(Path(input_video_path).resolve())
		cap = cv2.VideoCapture(input_video_path)
		frames_data=[]

		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		prev_horse_box = None

		with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
			while cap.isOpened():
				ret, frame = cap.read()
				if not ret:
					break

				result = self.model.predict(frame)

				if result[0].keypoints is not None and result[0].boxes.cls is not None:
					cls_array = result[0].boxes.cls.cpu().numpy()  # Convert class tensor to numpy
					if len(cls_array) > 0:  # Ensure non-empty detections
						horse_keypoints = result[0].keypoints[cls_array == 0].xy.cpu().numpy()
						horse_boxes = result[0].boxes[cls_array == 0].xyxy.cpu().numpy()
						if prev_horse_box is not None:
							ious = [compute_iou(prev_horse_box, box) for box in horse_boxes]
							max_iou_idx = np.argmax(ious)

							horse_keypoints = [horse_keypoints[max_iou_idx]]
							horse_boxes = [horse_boxes[max_iou_idx]]
						prev_horse_box = horse_boxes[0] if len(horse_boxes)>0 else None

						for keypoints, box in zip(horse_keypoints, horse_boxes):
							if not((keypoints[18,0]==0 and keypoints[18,1]==0) and (keypoints[19,0]==0 and keypoints[19,1]==0)):
								if (keypoints[18,0]==0 and keypoints[18,1]==0):
									keypoints[18, 0]=  keypoints[11, 0]
									keypoints[18, 1]= keypoints[19, 1]
								elif (keypoints[19,0]==0 and keypoints[19,1]==0):
									keypoints[19, 0]=  keypoints[10, 0]
									keypoints[19, 1]= keypoints[18, 1]

							frames_data.append({'keypoints':keypoints.tolist(), 'bbox': box.tolist()})
					else:
						keypoints = [[0,0], [0,0], [0,0], [0,0] ,[0,0], [0,0], [0,0], [0,0], [0,0], [0,0],
									 [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
						box = None
						frames_data.append({'keypoints':keypoints, 'bbox': box })

				else:
					keypoints = [[0,0], [0,0], [0,0], [0,0] ,[0,0], [0,0], [0,0], [0,0], [0,0], [0,0],
								 [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
					box = None
					frames_data.append({'keypoints':keypoints, 'bbox': box })



				pbar.update(1)


		cap.release()
		with open(self.output_file_path, "w") as json_file:
			json.dump(frames_data, json_file,  indent=4)
			json_file.flush()
			os.fsync(json_file.fileno())
		print("Pose estimation output data are saved!")


	def process_video_v1(self, input_video_path, output_video_path ):

		input_video_path = str(Path(input_video_path).resolve())
		output_video_path = str(Path(output_video_path).resolve())

		cap = cv2.VideoCapture(input_video_path)

		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = cap.get(cv2.CAP_PROP_FPS)

		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

		prev_horse_box = None

		with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
			while cap.isOpened():
				ret, frame = cap.read()
				if not ret:
					break

				result = self.model.predict(frame)

				if result[0].keypoints is not None and result[0].boxes.cls is not None:
					cls_array = result[0].boxes.cls.cpu().numpy()
					if len(cls_array) > 0:
						horse_keypoints = result[0].keypoints[cls_array == 0].xy.cpu().numpy()
						horse_boxes = result[0].boxes[cls_array == 0].xyxy.cpu().numpy()
						if prev_horse_box is not None:
							ious = [compute_iou(prev_horse_box, box) for box in horse_boxes]
							max_iou_idx = np.argmax(ious)
							horse_keypoints = [horse_keypoints[max_iou_idx]]
							horse_boxes = [horse_boxes[max_iou_idx]]
						prev_horse_box = horse_boxes[0] if len(horse_boxes)>0 else None

						for keypoints in horse_keypoints:
							if not((keypoints[18,0]==0 and keypoints[18,1]==0) and (keypoints[19,0]==0 and keypoints[19,1]==0)):
								if (keypoints[18,0]==0 and keypoints[18,1]==0):
									keypoints[18, 0]=  keypoints[11, 0]
									keypoints[18, 1]= keypoints[19, 1]
								elif (keypoints[19,0]==0 and keypoints[19,1]==0):
									keypoints[19, 0]=  keypoints[10, 0]
									keypoints[19, 1]= keypoints[18, 1]
							self.draw_horse_skeleton(frame, keypoints)

				out.write(frame)

				pbar.update(1)


		cap.release()
		out.release()
		print(f'Video {input_video_path} is processed and saved as {output_video_path}')




