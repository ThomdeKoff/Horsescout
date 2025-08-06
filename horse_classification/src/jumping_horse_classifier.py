import numpy as np
import joblib

from .utils import load_config
from .horse_feature_extractor import HorseFeatureExtractor

class JumpingHorseClassifier:
	def __init__(self, fps):
		config = load_config()
		self.model = joblib.load(config['paths']['rf_jumping_classification_model'])
		self.scaler = joblib.load(config['paths']['rf_jumping_classification_scaler'])
		self.feature_extractor= HorseFeatureExtractor(fps = fps)


	def classify(self, preprocessed_data, bboxes):

		features = self.feature_extractor.extract_general_features(preprocessed_data, bboxes)
		features.extend(self.feature_extractor.extract_jumping_features(preprocessed_data))
		features = np.nan_to_num(self.scaler.transform([features]))
		prediction = self.model.predict(features)
		prediction_prob = self.model.predict_proba(features)

		margin = prediction_prob[0][0] - prediction_prob[0][1]
		if 0 < margin <= 0.12:
			prediction[0] = 1  # override as "good"

		return prediction
