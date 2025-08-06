import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import find_peaks


from .utils import load_config


class HorseFeatureExtractor:
	def __init__(self, fps):
		config=load_config()
		self.KEYPOINTS_ORDER = config['keyjoints_configuration']['keyjoints_order']
		self.fps = fps

	def extract_general_features(self, keypoints_array, bboxes_array=None):
		"""
        Extracts features from a sequence of keypoints and bounding boxes.

        Parameters:
            keypoints_array: numpy array [num_frames, num_keypoints, 2] - processed keypoints
            bboxes_array: numpy array [num_frames, 4] - each bbox is [xc, yc, w, h]

        Returns:
            feature_vector: 1D numpy array of extracted features
        """

		features = []

		# --- Helper: compute distance between keypoints ---
		def compute_dist(kp1_idx, kp2_idx):
			kp1 = keypoints_array[:, kp1_idx, :]
			kp2 = keypoints_array[:, kp2_idx, :]
			dist = np.linalg.norm(kp1 - kp2, axis=1)  # Shape: [num_frames]
			return dist

		# --- 1. Relative Distances (mean/std/max/min) ---
		flank_idx = self.KEYPOINTS_ORDER.index('flank')
		head_idx = self.KEYPOINTS_ORDER.index('forehead')
		front_hoof_r = self.KEYPOINTS_ORDER.index('frontHoof right')
		front_hoof_l = self.KEYPOINTS_ORDER.index('frontHoof left')
		back_hoof_r = self.KEYPOINTS_ORDER.index('backHoof right')
		back_hoof_l = self.KEYPOINTS_ORDER.index('backHoof left')

		dist_head_flank = compute_dist(head_idx, flank_idx)
		dist_front_hoof_flank = (compute_dist(front_hoof_r, flank_idx) + compute_dist(front_hoof_l, flank_idx)) / 2
		dist_back_hoof_flank = (compute_dist(back_hoof_r, flank_idx) + compute_dist(back_hoof_l, flank_idx)) / 2

		for dist in [dist_head_flank, dist_front_hoof_flank, dist_back_hoof_flank]:
			if len(dist) == 0 or np.isnan(dist).all():
				features += [0, 0, 0, 0]
			else:
				features += [np.nanmean(dist), np.nanstd(dist), np.nanmin(dist), np.nanmax(dist)]

		# --- 2. Joint Velocities ---
		velocities = np.diff(keypoints_array, axis=0)  # shape: [num_frames-1, num_keypoints, 2]
		vel_magnitudes = np.linalg.norm(velocities, axis=2)  # [num_frames-1, num_keypoints]
		mean_vel = np.nanmean(vel_magnitudes, axis=0)  # [num_keypoints]
		std_vel = np.nanstd(vel_magnitudes, axis=0)

		features += list(mean_vel) + list(std_vel)  # Append per-joint velocity stats

		# --- 3. Joint Angles (front legs and back legs) ---
		def compute_angle(a_idx, b_idx, c_idx):
			a = keypoints_array[:, a_idx, :]
			b = keypoints_array[:, b_idx, :]
			c = keypoints_array[:, c_idx, :]

			ba = a - b
			bc = c - b

			# Normalize
			ba_norm = ba / np.linalg.norm(ba, axis=1, keepdims=True)
			bc_norm = bc / np.linalg.norm(bc, axis=1, keepdims=True)

			dot_product = np.einsum('ij,ij->i', ba_norm, bc_norm)
			angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Radians
			angle_deg = np.degrees(angle)
			return angle_deg  # [num_frames]

		# Example angles: shoulder-flank-hip (right side)
		shoulder_r = self.KEYPOINTS_ORDER.index('shoulder right')
		back_elbow_r = self.KEYPOINTS_ORDER.index('backElbow right')

		angle_shoulder_flank_hip = compute_angle(shoulder_r, flank_idx, back_elbow_r)
		features += [np.nanmean(angle_shoulder_flank_hip), np.nanstd(angle_shoulder_flank_hip)]
		if bboxes_array:
			# --- 4. BBox Dynamics (jump arc) ---
			yc=[]
			height=[]

			for bbox in bboxes_array:
				if bbox:
					yc.append((bbox[1]+bbox[3])/2)
					height.append(bbox[3]-bbox[1])
				else:
					yc.append(np.nan)
					height.append(np.nan)
			features += [np.nanmax(yc), np.nanmin(yc), np.nanmean(yc), np.nanstd(yc)]
			features += [np.nanmax(height), np.nanmean(height), np.nanstd(height)]

		return features

	def extract_general_features_v1(self, keypoints_array, bboxes_array=None):

		num_frames = keypoints_array.shape[0]
		features = []

		# --- Helper: compute distance between keypoints ---
		def compute_dist(kp1_idx, kp2_idx):
			kp1 = keypoints_array[:, kp1_idx, :]
			kp2 = keypoints_array[:, kp2_idx, :]
			dist = np.linalg.norm(kp1 - kp2, axis=1)  # Shape: [num_frames]
			return dist

		# --- 1. Relative Distances (mean/std/max/min) ---
		flank_idx = self.KEYPOINTS_ORDER.index('flank')
		head_idx = self.KEYPOINTS_ORDER.index('forehead')
		front_hoof_r = self.KEYPOINTS_ORDER.index('frontHoof right')
		front_hoof_l = self.KEYPOINTS_ORDER.index('frontHoof left')
		back_hoof_r = self.KEYPOINTS_ORDER.index('backHoof right')
		back_hoof_l = self.KEYPOINTS_ORDER.index('backHoof left')

		dist_head_flank = compute_dist(head_idx, flank_idx)
		dist_front_hoof_flank = (compute_dist(front_hoof_r, flank_idx) + compute_dist(front_hoof_l, flank_idx)) / 2
		dist_back_hoof_flank = (compute_dist(back_hoof_r, flank_idx) + compute_dist(back_hoof_l, flank_idx)) / 2

		for dist in [dist_head_flank, dist_front_hoof_flank, dist_back_hoof_flank]:
			if len(dist) == 0 or np.isnan(dist).all():
				features += [0, 0, 0, 0]
			else:
				features += [np.nanmean(dist), np.nanstd(dist), np.nanmin(dist), np.nanmax(dist)]

		# --- 2. Joint Velocities ---
		velocities = np.diff(keypoints_array, axis=0)  # shape: [num_frames-1, num_keypoints, 2]
		vel_magnitudes = np.linalg.norm(velocities, axis=2)  # [num_frames-1, num_keypoints]
		mean_vel = np.nanmean(vel_magnitudes, axis=0)  # [num_keypoints]
		std_vel = np.nanstd(vel_magnitudes, axis=0)

		features += list(mean_vel) + list(std_vel)  # Append per-joint velocity stats

		# --- 3. Joint Angles (front legs and back legs) ---
		def compute_angle(a_idx, b_idx, c_idx):
			a = keypoints_array[:, a_idx, :]
			b = keypoints_array[:, b_idx, :]
			c = keypoints_array[:, c_idx, :]

			ba = a - b
			bc = c - b

			# Normalize
			ba_norm = ba / np.linalg.norm(ba, axis=1, keepdims=True)
			bc_norm = bc / np.linalg.norm(bc, axis=1, keepdims=True)

			dot_product = np.einsum('ij,ij->i', ba_norm, bc_norm)
			angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Radians
			angle_deg = np.degrees(angle)
			return angle_deg  # [num_frames]

		# Example angles: shoulder-flank-hip (right side)
		shoulder_r = self.KEYPOINTS_ORDER.index('shoulder right')
		back_elbow_r = self.KEYPOINTS_ORDER.index('backElbow right')

		angle_shoulder_flank_hip = compute_angle(shoulder_r, flank_idx, back_elbow_r)
		features += [np.nanmean(angle_shoulder_flank_hip), np.nanstd(angle_shoulder_flank_hip)]
		features.append(num_frames)
		return features

	def extract_stride_features(self, keypoints_array):
		"""
        Extracts stride-based motion features from normalized and local coordinate keypoints.

        Parameters:
            segment: numpy array of shape [num_frames, num_keypoints, 2]
            fps: Frames per second of the video (used for frequency calculations)

        Returns:
            A dictionary with extracted features:
                - Stride frequency (Hz)
                - Average stride length
                - Step symmetry
                - Hoof height variation
        """
		FRONT_HOOF_RIGHT_IDX = self.KEYPOINTS_ORDER.index("frontHoof right")
		FRONT_HOOF_LEFT_IDX = self.KEYPOINTS_ORDER.index("frontHoof left")

		# Extract front hooves coordinates
		front_hoof_right = keypoints_array[:, FRONT_HOOF_RIGHT_IDX, :]  # (num_frames, 2)
		front_hoof_left = keypoints_array[:, FRONT_HOOF_LEFT_IDX, :]    # (num_frames, 2)

		# Y-coordinates for vertical motion analysis
		y_right = front_hoof_right[:, 1]
		y_left = front_hoof_left[:, 1]

		# Detect peaks (highest point in stride cycle)
		peaks_right, _ = find_peaks(-y_right, distance=self.fps//2)  # Negative since upward motion is -Y
		peaks_left, _ = find_peaks(-y_left, distance=self.fps//2)

		# Compute Stride Frequency (Hz) = (number of strides) / (time in seconds)
		num_strides = max(len(peaks_right), len(peaks_left))
		stride_frequency = num_strides / (len(keypoints_array) / self.fps) if len(keypoints_array) > 0 else 0

		# Compute Stride Length (average horizontal displacement between peaks)
		if len(peaks_right) > 1:
			stride_length_right = np.mean(np.abs(np.diff(front_hoof_right[peaks_right, 0])))
		else:
			stride_length_right = 0

		if len(peaks_left) > 1:
			stride_length_left = np.mean(np.abs(np.diff(front_hoof_left[peaks_left, 0])))
		else:
			stride_length_left = 0

		avg_stride_length = (stride_length_right + stride_length_left) / 2

		# Step Symmetry (difference in peak timing)
		step_symmetry = np.abs(len(peaks_right) - len(peaks_left)) / max(len(peaks_right), len(peaks_left), 1)

		# Hoof Height Variation (difference between highest and lowest points)
		hoof_height_variation = np.max(y_right) - np.min(y_right) + np.max(y_left) - np.min(y_left)

		return (
			stride_frequency,
			avg_stride_length,
			step_symmetry,
			hoof_height_variation
		)
	def extract_jumping_features(self, keypoints_array):
		"""
		Extracts features specifically related to jumping.

		Parameters:
			keypoints_array: numpy array [num_frames, num_keypoints, 2] - processed keypoints

		Returns:
			feature_vector: 1D numpy array of extracted features
		"""
		num_frames = keypoints_array.shape[0]
		features = []

		# --- Helper: compute distance between keypoints ---
		def compute_dist(kp1_idx, kp2_idx):
			kp1 = keypoints_array[:, kp1_idx, :]
			kp2 = keypoints_array[:, kp2_idx, :]
			dist = np.linalg.norm(kp1 - kp2, axis=1)  # Shape: [num_frames]
			return dist

		# --- Keypoints Indices ---
		flank_idx = self.KEYPOINTS_ORDER.index('flank')
		front_hoof_r = self.KEYPOINTS_ORDER.index('frontHoof right')
		front_hoof_l =self.KEYPOINTS_ORDER.index('frontHoof left')
		back_hoof_r = self.KEYPOINTS_ORDER.index('backHoof right')
		back_hoof_l = self.KEYPOINTS_ORDER.index('backHoof left')

		# --- 1. Detect Takeoff & Landing ---
		def detect_takeoff_landing(hoof_idx):
			"""Find the frame indices of takeoff and landing based on hoof Y movement."""
			y_positions = keypoints_array[:, hoof_idx, 1]  # Extract Y-coordinates
			y_positions = y_positions[~np.isnan(y_positions)]  # Filter NaNs

			if y_positions.size == 0:
				return 0, 0, np.nan, np.nan  # or handle differently based on context

			min_y = np.nanmin(y_positions)
			max_y = np.nanmax(y_positions)

			full_y = keypoints_array[:, hoof_idx, 1]
			threshold = min_y + 0.1 * (max_y - min_y)

			takeoff_idx_candidates = np.where(full_y > threshold)[0]
			if len(takeoff_idx_candidates) == 0:
				return 0, 0, min_y, max_y
			takeoff_idx = takeoff_idx_candidates[0]

			landing_candidates = np.where(full_y[takeoff_idx:] < threshold)[0]
			if len(landing_candidates) == 0:
				landing_idx = takeoff_idx
			else:
				landing_idx = takeoff_idx + landing_candidates[0]

			return takeoff_idx, landing_idx, min_y, max_y

		takeoff_r, landing_r, min_r, max_r = detect_takeoff_landing(front_hoof_r)
		takeoff_l, landing_l, min_l, max_l = detect_takeoff_landing(front_hoof_l)

		# Average across hooves
		avg_takeoff_frame = (takeoff_r + takeoff_l) / 2
		avg_landing_frame = (landing_r + landing_l) / 2
		avg_min_y = (min_r + min_l) / 2
		avg_max_y = (max_r + max_l) / 2

		jump_duration = avg_landing_frame - avg_takeoff_frame  # Number of frames in air

		features += [jump_duration, avg_min_y, avg_max_y]

		# --- 2. Hoof Trajectory & Distance ---
		def compute_hoof_trajectory(hoof_idx):
			x_vals = keypoints_array[:, hoof_idx, 0]
			y_vals = keypoints_array[:, hoof_idx, 1]

			# Remove NaNs
			x_vals = x_vals[~np.isnan(x_vals)]
			y_vals = y_vals[~np.isnan(y_vals)]

			if x_vals.size == 0 or y_vals.size == 0:
				return 0.0, 0.0  # or np.nan, np.nan depending on what you want

			x_travel = np.nanmax(x_vals) - np.nanmin(x_vals)
			y_travel = np.nanmax(y_vals) - np.nanmin(y_vals)
			return x_travel, y_travel


		traj_r_x, traj_r_y = compute_hoof_trajectory(front_hoof_r)
		traj_l_x, traj_l_y = compute_hoof_trajectory(front_hoof_l)

		features += [traj_r_x, traj_r_y, traj_l_x, traj_l_y]

		# --- 3. Jump Velocity (Hoof Speed) ---
		def compute_hoof_velocity(hoof_idx):
			coords = keypoints_array[:, hoof_idx, :]
			coords = coords[~np.isnan(coords).any(axis=1)]  # Remove frames with NaN for this hoof

			if coords.shape[0] < 2:
				return 0.0, 0.0  # Not enough points to compute velocity

			velocities = np.diff(coords, axis=0)  # [num_frames-1, 2]
			speed = np.linalg.norm(velocities, axis=1)  # [num_frames-1]

			if speed.size == 0:
				return 0.0, 0.0

			return np.nanmean(speed), np.nanmax(speed)


		avg_vel_r, max_vel_r = compute_hoof_velocity(front_hoof_r)
		avg_vel_l, max_vel_l = compute_hoof_velocity(front_hoof_l)

		features += [avg_vel_r, max_vel_r, avg_vel_l, max_vel_l]

		# --- 4. Symmetry of Jump ---
		symmetry_x = abs(traj_r_x - traj_l_x)
		symmetry_y = abs(traj_r_y - traj_l_y)

		features += [symmetry_x, symmetry_y]

		return features
