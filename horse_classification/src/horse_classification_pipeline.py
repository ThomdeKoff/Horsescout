import numpy as np
import os
import json

from .utils import load_config
from .keypoint_preprocessor import KeypointPreprocessor
from .jumping_horse_classifier import JumpingHorseClassifier
from .running_horse_classifier import  RunningHorseClassifier

class HorsePipeline:
	def __init__(self, fps):
		self.jumping_classifier = JumpingHorseClassifier(fps = fps)
		self.running_classifier = RunningHorseClassifier(fps = fps)
		self.keyjoints_preprocessor = KeypointPreprocessor()

	def run(self):
		config = load_config()

		with open(config['paths']['pose_estimation_output'], 'r') as file:
			frames_data = json.load(file)

		with open(config['paths']['movement_recognition_output'], 'r') as file:
			predictions = json.load(file)

		results = []
		for best_prob, movement, start, end in predictions:
			segment = frames_data[start:end]
			data, bboxes= self.keyjoints_preprocessor.run_preprocessing_pipeline_v1(frames_data=segment)

			if movement == "Jumping":
				pred = self.jumping_classifier.classify(data, bboxes)
				results.append(pred)


		# else:
		# 	pred = self.running_classifier.classify(data)



		avg_score = np.mean(np.vstack(results), axis=0)
		if avg_score <= 0.5:
			label = "bad horse"
			conf =  1 - avg_score

		else:
			label = "good horse"
			conf = avg_score
		with open(config['paths']['classification_output'], "w") as json_file:
			json.dump({"label": label, "conf": conf[0]}, json_file,  indent=4)
			json_file.flush()
			os.fsync(json_file.fileno())

		return label, conf
