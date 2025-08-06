import numpy as np
from scipy.signal import savgol_filter


from .utils import  load_config


class KeypointPreprocessor:
	def __init__(self):
		config = load_config()
		self.keypoints_order = config['keyjoints_configuration']['keyjoints_order']
		self.flank_idx = self.keypoints_order.index("flank")
		self.forehead_idx = self.keypoints_order.index("forehead")
		self.dock_idx = self.keypoints_order.index("dock")

	def preprocess_frames(self, frames_data):
		num_frames = len(frames_data)
		num_keypoints = len(frames_data[0]['keypoints'])
		keypoints_array = np.full((num_frames, num_keypoints, 2), np.nan, dtype=np.float32)

		for i, frame in enumerate(frames_data):
			keypoints = np.array(frame['keypoints'], dtype=np.float32)
			keypoints[keypoints == 0] = np.nan
			keypoints_array[i] = keypoints

		return keypoints_array

	def normalize(self, segment, range_01=True):
		min_vals = np.nanmin(segment, axis=(0, 1), keepdims=True)
		max_vals = np.nanmax(segment, axis=(0, 1), keepdims=True)
		denom = max_vals - min_vals
		denom[denom == 0] = 1

		if range_01:
			return (segment - min_vals) / denom
		else:
			return 2 * (segment - min_vals) / denom - 1

	def smooth(self, keypoints, window=5, poly=2):
		smoothed = np.copy(keypoints)
		num_frames, num_keypoints, _ = keypoints.shape

		for j in range(num_keypoints):
			for dim in range(2):
				smoothed[:, j, dim] = savgol_filter(
					keypoints[:, j, dim], window, poly, mode='nearest'
				)

		return smoothed

	def translate_to_local(self, segment, smooth=True):
		flank = segment[:, self.flank_idx, :]
		forehead = segment[:, self.forehead_idx, :]
		dock = segment[:, self.dock_idx, :]

		valid_frames = ~np.isnan(flank).any(axis=1)
		valid_frames &= (~np.isnan(forehead).any(axis=1)) | (~np.isnan(dock).any(axis=1))

		segment = segment[valid_frames]
		flank = flank[valid_frames]
		forehead = forehead[valid_frames]
		dock = dock[valid_frames]

		x_vec = np.zeros_like(flank)
		for i in range(segment.shape[0]):
			if not np.isnan(forehead[i]).any():
				x_vec[i] = forehead[i] - flank[i]
			else:
				x_vec[i] = flank[i] - dock[i]

		x_vec /= np.linalg.norm(x_vec, axis=1, keepdims=True)
		y_vec = np.stack([-x_vec[:, 1], x_vec[:, 0]], axis=1)

		translated = segment - flank[:, np.newaxis, :]

		transformed_x = np.einsum("fkp,fp->fk", translated, x_vec)
		transformed_y = np.einsum("fkp,fp->fk", translated, y_vec)
		transformed = np.stack([transformed_x, transformed_y], axis=2)

		if smooth:
			transformed = self.smooth(transformed)

		return transformed

	def run_preprocessing_pipeline(self, frames_data):
		preprocessed_data= self.preprocess_frames(frames_data)
		normalized_data = self.normalize(preprocessed_data)
		translated_data = self.translate_to_local(normalized_data)
		return translated_data

	def run_preprocessing_pipeline_v1(self, frames_data):
		bboxes=[]
		preprocessed_data= self.preprocess_frames(frames_data)
		normalized_data = self.normalize(preprocessed_data)
		translated_data = self.translate_to_local(normalized_data)
		for frame in frames_data:
			bboxes.append(frame['bbox'])
		return translated_data, bboxes