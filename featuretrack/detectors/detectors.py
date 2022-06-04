import numpy as np
from cv2 import cv2 as cv
from abc import ABC, abstractmethod
from featuretrack.detectors.utils import Point

class FeatureExtractor(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def get_feature_points(self, frame):
    pass

  @abstractmethod
  def get_features(self, frame):
    pass

class SIFTFE(FeatureExtractor):
  def __init__(self, num_features):
    super(SIFTFE, self).__init__()
    self.num_features = num_features
    self.sift = cv.SIFT_create(self.num_features)

  def get_feature_points(self, frame):
    keypoints, _ = self.get_features(frame)
    return self._keypoints_to_tuple_list(keypoints)

  def get_features(self, frame):
    sift = cv.SIFT_create(self.num_features)
    return sift.detectAndCompute(frame, None)

  def _keypoints_to_tuple_list(self, keypoints):
    return np.array([ Point.from_SIFT(keypoint).to_tuple() for keypoint in keypoints ], dtype=np.float32)


class GoodFeaturesToTrackFE(FeatureExtractor):
  def __init__(self, max_corners, quality_level, min_distance, block_size):
    super(GoodFeaturesToTrackFE, self).__init__()
    self.max_corners = max_corners
    self.quality_level = quality_level
    self.min_distance = min_distance
    self.block_size = block_size

  def get_feature_points(self, frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    points = self.get_features(frame_gray)
    return np.array([ Point.from_GFT(point).to_tuple() for point in points ], dtype=np.float32)

  def get_features(self, frame_gray):
    return cv.goodFeaturesToTrack(
      frame_gray,
      maxCorners = self.max_corners,
      qualityLevel = self.quality_level,
      minDistance = self.min_distance,
      blockSize = self.block_size,
      useHarrisDetector = False,
    )
