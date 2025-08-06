import argparse

from horse_segmentation import HorseSegmentation

def main():
	parser = argparse.ArgumentParser(
		description="Horse segmentation: Video processing")
	parser.add_argument('--input_video_path', type=str,
						help="Path to the input video")

	parser.add_argument('--output_video_path', type=str,
						help="Path where to save the output video")

	args = parser.parse_args()
	horse_segmentation = HorseSegmentation()
	horse_segmentation.process_video(input_video_path=args.input_video_path, output_video_path=args.output_video_path)

if __name__ == '__main__':
	main()