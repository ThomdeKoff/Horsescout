import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import logging

class HorseSegmentation:

	def __init__(self, model_path: str = 'yolo11n-seg.pt'):
		logging.getLogger("ultralytics").setLevel(logging.WARNING)
		self.model = YOLO(str(Path(model_path).resolve()))

	def process_video(self, input_video_path: str, output_video_path: str) -> None:
		"""
		Processes the video frame by frame to segment the horse and visualize the segmented area by overlaying a green mask
		:param input_video_path: the path to the input video
		:param output_video_path:  the path where to save the output video
		"""
		input_video_path = Path(input_video_path).resolve()
		output_video_path = Path(output_video_path).resolve()

		cap = cv2.VideoCapture(str(input_video_path))

		fps = int(cap.get(cv2.CAP_PROP_FPS))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:

			while cap.isOpened():
				ret, frame = cap.read()
				if not ret:
					break

				# Run YOLO segmentation on the single frame
				result = self.model(frame, retina_masks=True)[0]

				# Extract horse masks (assuming class 17 corresponds to horses)
				horse_masks = result.masks[result.boxes.cls == 17] if result.masks is not None else []

				mask_frame = np.zeros((height, width, 3), dtype=np.uint8)

				for horse_mask in horse_masks:
					mask = horse_mask.data.squeeze().cpu().numpy() * 255
					mask = mask.astype(np.uint8)

					mask = cv2.resize(mask, (width, height))

					green_mask = np.zeros((height, width, 3), dtype=np.uint8)
					green_mask[:, :, 1] = mask

					mask_frame = cv2.bitwise_or(mask_frame, green_mask)

				masked_frame = cv2.addWeighted(frame, 0.7, mask_frame, 0.3, 0)

				out.write(masked_frame)

				pbar.update(1)


		cap.release()
		out.release()
		print(f'Video {input_video_path} is processed and saved as {output_video_path}')
