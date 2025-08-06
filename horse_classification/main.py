import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import cv2

from src.pose_estimation import HorsePoseEstimation
from src.horse_movement_recognition import HorseMovementRecognition
from src.horse_classification_pipeline import HorsePipeline
from src.utils import get_video_fps_height, reduce_video_resolution_1080p, load_config
from src.visualizer import Visualizer

def main():
	parser = argparse.ArgumentParser(description="Horse classification")
	parser.add_argument('--input_video_path', type=str, required=True, help="Path to the input video")
	parser.add_argument('--no-visualize', dest='visualize', action='store_false', help='Disable annotated video')
	parser.set_defaults(visualize=True)
	args = parser.parse_args()

	if not args.input_video_path:
		print("Error: Please provide a valid video path.")
		return

	fps, height = get_video_fps_height(video_path=args.input_video_path)
	if not fps:
		print("Error: Could not open the video.")
		return
	video_path = args.input_video_path
	if height > 1080:
		config = load_config()
		video_path = config['paths']['converted_video']
		reduce_video_resolution_1080p(input_video_path=args.input_video_path, output_video_path=video_path)
		
	print("Video processing: pose estimation...")
	HorsePoseEstimation().process_video(input_video_path=video_path)

	print("Horse movements recognition...")
	HorseMovementRecognition(fps=fps).predict()

	print("Horse data analysis...")
	label, (confidence,) = HorsePipeline(fps=fps).run()

	print(f"The horse is classified as '{label}'")

	if args.visualize:
		print("Building the annotated video...")
		Visualizer().run(input_video_path=video_path)

if __name__ == '__main__':
	main()
