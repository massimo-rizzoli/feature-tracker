import argparse
from argparse import ArgumentParser
from featuretrack.trackers.algorithms import lucas_kanade, sift, kalman


def add_gft_sift_detectors_subparsers(parser: ArgumentParser):
  subparsers = parser.add_subparsers(help='Feature extraction algorithm')

  gft_parser = subparsers.add_parser(
    'gft', help='Extract feature points according to the Good Features To Track algorithm',
    description='',
    epilog=''
  )
  gft_parser.set_defaults(features='gft')
  gft_parser.add_argument('-p', '--points', default=100, type=int,
                          help='Maximum number of corners to extract [default: 100]')
  gft_parser.add_argument('-q', '--quality', default=0.2, type=float,
                          help='Minimum accepted quality ratio with respect to the highest quality corner [default: 0.2]')
  gft_parser.add_argument('-d', '--distance', default=10, type=int,
                          help='Minimum euclidian distance between corners [default: 10]')
  gft_parser.add_argument('-s', '--size', default=3, type=int,
                          help='Block size around pixels for computing features [default: 3]')

  sift_parser = subparsers.add_parser(
    'sift', help='Extract feature points according to the Scale-Invariant Feature Transform algorithm',
    description='',
    epilog=''
  )
  sift_parser.set_defaults(features='sift')
  sift_parser.add_argument('-p', '--points', default=100, type=int, help='Number of keypoints to extract [default: 100]')


def add_lucas_kanade_subparser(subparsers):
  lk_parser = subparsers.add_parser(
    'lk', help='Lucas-Kanade tracking algorithm',
    description='Lucas-Kanade tracking, working with feature points extracted by the specified algorithm',
    epilog='')
  lk_parser.set_defaults(func=lucas_kanade)
  lk_parser.add_argument('-i', '--interval', default=60, type=int,
                         help='Interval of frames in which feature points are tracked before being recomputed [default: 60]')
  add_gft_sift_detectors_subparsers(lk_parser)
  return lk_parser


def add_sift_subparser(subparsers):
  sift_parser = subparsers.add_parser(
    'sift', help='Tracking based on L2-distance between SIFT point descriptors',
    description='Feature points are periodically computed by the Scale-Invariant Feature Transform algorithm and at each frame they are tracked by matching them with the most similar new points based on L2-distance between original and new SIFT descriptors',
    epilog=''
  )
  sift_parser.set_defaults(func=sift)
  sift_parser.set_defaults(features='sift')
  sift_parser.add_argument('-p', '--points', default=100, type=int, help='Number of keypoints to extract [default: 100]')
  sift_parser.add_argument('-i', '--interval', default=60, type=int,
                           help='Interval of frames in which feature points are tracked before being recomputed [default: 60]')
  sift_parser.add_argument('-t', '--thresh', default=90, type=int,
                           help='Matching quality threshold (distance) [default: 90]')
  return sift_parser


def add_kalman_subparser(subparsers):
  kalman_parser = subparsers.add_parser(
    'kalman', help='Kalman Filter tracking algorithm',
    description='Kalman Filter tracking, working with feature points extracted by the specified algorithm',
    epilog='')
  kalman_parser.set_defaults(func=kalman)
  kalman_parser.add_argument('-i', '--interval', default=60, type=int,
                             help='Interval of frames in which feature points are tracked before being recomputed [default: 60]')
  kalman_parser.add_argument('-c', '--correct', default=1, type=int,
                             help='Number of frames after which ground truth information is used to update the prediction of the Kalman Filter [default: 1]')
  kalman_parser.add_argument('-a', '--avgacc', default=False, action='store_true',
                             help='If specified, use the average accelerations of the previous feature points when recomputing the feature points, otherwise set the accelerations of the new feature points to zero')
  add_gft_sift_detectors_subparsers(kalman_parser)
  return kalman_parser


def prepare_parser():
  parser = argparse.ArgumentParser(description='Track features using different tracking algorithms combined with different feature detectors',
                                   prog='python -m featuretrack')
  # TODO remove default
  parser.add_argument('-v', '--video', default='./source_videos/Contesto_industriale1.mp4', type=str,
                      help='Path to the video on which to perform tracking')
  parser.add_argument('-f', '--framerate', default=60, type=int, help='Video framerate [default: 60]')
  parser.add_argument('--scale', nargs=2, default=[1,1], type=float, metavar=('H_SCALE', 'W_SCALE'),
                      help='Scaling factors to be applyed respectively to the height and the width of the video [default: 1 1]')
  parser.add_argument('-o','--output', type=str,
                      help='Output file name (e.g. \'file\' will be saved as \'./results/file.avi\') [default: generated from arguments]')
  parser.add_argument('-r', '--radious', default=3, type=int,
                         help='Radious of circles used to show the location of feature points [default: 3]')
  parser.add_argument('-t', '--thickness', default=2, type=int,
                         help='Thickness of the circle borders [default: 2]')
  subparsers = parser.add_subparsers(help='Tracking algorithm')

  add_lucas_kanade_subparser(subparsers)

  add_sift_subparser(subparsers)

  add_kalman_subparser(subparsers)

  return parser
