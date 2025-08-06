import numpy as np
import joblib

from .utils import load_config
from .horse_feature_extractor import HorseFeatureExtractor


class RunningHorseClassifier:
	def __init__(self,  fps):
		config = load_config()
		self.model = joblib.load(config['paths']['running_horse_classification_model'])
		self.scaler = joblib.load(config['paths']['running_horse_classification_scaler'])
		self.feature_extractor = HorseFeatureExtractor(fps = fps)

	def classify(self, preprocessed_data):

		features = self.feature_extractor.extract_general_features(preprocessed_data)
		stride_features = self.feature_extractor.extract_stride_features(preprocessed_data)
		features.extend(stride_features)

		features = np.nan_to_num(self.scaler.transform([features]))

		prediction = self.model.predict(features)
		probs = self.model.predict_proba(features)[0]

		margin = probs[1] - probs[0]
		if 0 < margin <= 0.1:
			prediction[0] = 0  # classify as "bad"

		return prediction