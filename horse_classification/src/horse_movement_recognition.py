from math import nan
import numpy as np
import json
import os
import joblib

from .utils import  load_config
from .keypoint_preprocessor import  KeypointPreprocessor
from .horse_feature_extractor import HorseFeatureExtractor

class HorseMovementRecognition:
	def __init__(self, fps):
		config= load_config()
		self.clf = joblib.load(config['paths']['movement_recognition_model'])
		self.scaler = joblib.load(config['paths']['movement_recognition_scaler'])
		self.window_size = config['sliding_window']['window_size']
		self.step_size = config['sliding_window']['stride']
		self.skip_threshold = config['sliding_window']['skip_threshold']
		self.prob_threshold = config['sliding_window']['probability_threshold']
		self.KEYPOINTS_ORDER = config['keyjoints_configuration']['keyjoints_order']
		self.output_file_path = config['paths']['movement_recognition_output']
		self.frames_data_path = config['paths']['pose_estimation_output']
		self.feature_extractor = HorseFeatureExtractor(fps = fps)
		self.keyjoints_preprocessor = KeypointPreprocessor()

	def check_keypoint_presence(self, frames):
		"""Skip window if flank & forehead are missing for many frames."""
		missing_count = 0
		flank_idx = self.KEYPOINTS_ORDER.index('flank')
		head_idx = self.KEYPOINTS_ORDER.index('forehead')

		for frame in frames:
			if frame['keypoints'][flank_idx] == [0, 0] and frame['keypoints'][head_idx] == [0, 0]:
				missing_count += 1
				if missing_count >= self.skip_threshold:
					return True
			else:
				missing_count = 0
		return False

	def process_and_predict(self, frames):
		"""Preprocess, extract features, scale, and predict."""
		try:
			processed_data = self.keyjoints_preprocessor.run_preprocessing_pipeline(frames_data=frames)
		except Exception as e:
			print("Horse movement recognition: preprocessing error: ",e)

		try:
			features = self.feature_extractor.extract_general_features(processed_data)
		except Exception as e:
			print("Horse movement recognition: feature extraction error: ",e)

		features = np.nan_to_num(self.scaler.transform([features]))
		return self.clf.predict_proba(features)[0]

	def predict(self):
		"""Run sliding window movement detection on full sequence."""
		with open(self.frames_data_path, 'r') as file:
			frames_data = json.load(file)

		total_frames = len(frames_data)
		predictions = []
		i = 0

		while i + self.window_size <= total_frames:
			if self.check_keypoint_presence(frames_data[i:i + self.window_size]):
				i += self.window_size
				continue

			best_prob = 0
			last_prob = None
			first_frame = i
			last_frame = i + self.window_size
			actual_prediction = ""

			for j in range(0, self.window_size, self.step_size):
				if i + self.window_size + j > total_frames:
					break

				current_frames = frames_data[i:i + self.window_size + j]
				probs = self.process_and_predict(current_frames)

				if probs[0] > probs[1]:
					max_prob = probs[0]
					prediction = "Running"
				else:
					max_prob = probs[1]
					prediction = "Jumping"

				if last_prob is not None and max_prob < last_prob - self.prob_threshold:
					break

				best_prob = max_prob
				last_prob = max_prob
				last_frame = i + self.window_size + j
				actual_prediction = prediction

			predictions.append((best_prob, actual_prediction, first_frame, last_frame))
			i += self.window_size

		with open(self.output_file_path, "w") as json_file:
			json.dump(predictions, json_file,  indent=4)
			json_file.flush()
			os.fsync(json_file.fileno())

		print("Horse movements data are saved!")


