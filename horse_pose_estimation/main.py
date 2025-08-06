import argparse

from horse_pose_estimation import HorsePoseEstimation

def main():
	parser = argparse.ArgumentParser(
		description="Horse pose estimation: Video processing")
	parser.add_argument('--input_video_path', type=str,
						help="Path to the input video")

	parser.add_argument('--output_video_path', type=str,
						help="Path where to save the output video")

	args = parser.parse_args()
	horse_pose_estimation= HorsePoseEstimation()
	horse_pose_estimation.process_video(input_video_path=args.input_video_path, output_video_path=args.output_video_path)

if __name__ == '__main__':
	main()