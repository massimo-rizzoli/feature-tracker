from cv2 import cv2 as cv
from featuretrack.parsers import prepare_parser
from featuretrack.detectors.detectors import GoodFeaturesToTrackFE, SIFTFE
from featuretrack.video_writer import VideoWriter

def filename_from_args(args):
  return 'test'

def main():
  parser = prepare_parser()

  args = parser.parse_args()

  # calculate and scale resolution
  tmp_cap = cv.VideoCapture(args.video)
  _, frame = tmp_cap.read()
  tmp_cap.release()
  args.resolution = (round(frame.shape[1]*args.scale[1]),round(frame.shape[0]*args.scale[0]))

  args.cap = cv.VideoCapture(args.video)

  # TODO check if formula correct
  args.delay = round(1/args.framerate * 1000)

  if args.output:
    args.writer = VideoWriter(''.join(['./results/', args.output]), args.framerate/2, args.resolution)
  else:
    args.writer = None

  if args.features == 'gft':
    args.fe = GoodFeaturesToTrackFE(args.points, args.quality, args.distance, args.size)
  elif args.features == 'sift':
    args.fe = SIFTFE(args.points)
  else:
    parser.print_usage()
    parser.exit(1)

  args.func(args)

if __name__ == '__main__':
  main()
