from cv2 import cv2 as cv
from os import path
from featuretrack.parsers import prepare_parser
from featuretrack.detectors.detectors import GoodFeaturesToTrackFE, SIFTFE
from featuretrack.video_writer import VideoWriter

def main():
  parser = prepare_parser()

  args = parser.parse_args()

  # calculate and scale resolution
  if not path.exists(args.video):
    print(f'Video path \'{args.video}\' does not exists')
    parser.exit(1)
  tmp_cap = cv.VideoCapture(args.video)
  _, frame = tmp_cap.read()
  tmp_cap.release()
  args.resolution = (round(frame.shape[1]*args.scale[1]),round(frame.shape[0]*args.scale[0]))
  writer_res = args.resolution
  # triplicate width for sift brute force windows
  if 'hidematch' in vars(args) and not args.hidematch:
    writer_res = (args.resolution[0]*3, args.resolution[1])

  args.cap = cv.VideoCapture(args.video)

  if args.output:
    args.writer = VideoWriter(''.join(['./results/', args.output]), args.framerate, writer_res)
  else:
    args.writer = None

  if 'features' not in vars(args):
    parser.print_help()
    exit(1)
  elif args.features == 'gft':
    args.fe = GoodFeaturesToTrackFE(args.points, args.quality, args.distance, args.size)
  elif args.features == 'sift':
    args.fe = SIFTFE(args.points)
  else:
    parser.print_help()
    parser.exit(1)

  args.func(args)

if __name__ == '__main__':
  main()
