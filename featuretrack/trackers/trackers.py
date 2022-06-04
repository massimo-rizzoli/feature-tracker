import numpy as np
from cv2 import cv2 as cv
from abc import ABC
from featuretrack.detectors.detectors import SIFTFE
from featuretrack.detectors.utils import Point

class Tracker(ABC):
  def __init__(self):
    pass


class LucasKanadeTracker(Tracker):
  def __init__(self):
    super(LucasKanadeTracker, self).__init__()

  def predict(self, prev_frame, frame, prev_corners):
    corners, status, err = cv.calcOpticalFlowPyrLK(prev_frame, frame, prev_corners, None)
    return corners


class SIFTTracker(Tracker):
  def __init__(self, num_features: int, match_threshold: int, sift_fe: SIFTFE):
    super(SIFTTracker, self).__init__()
    self.num_features = num_features
    self.matcher = cv.BFMatcher(cv.NORM_L2)
    self.sift_fe = sift_fe
    self.match_threshold = match_threshold

  def predict(self, prev_desc, frame):
    curr_kp, curr_desc = self.sift_fe.get_features(frame)
    matches = self.matcher.match(prev_desc, curr_desc)
    good_matches = list(filter(lambda x: x.distance < self.match_threshold, matches))
    return self.sift_fe._keypoints_to_tuple_list([ curr_kp[gm.trainIdx] for gm in good_matches ]), curr_kp, curr_desc, good_matches


class KalmanTracker(Tracker):
  def __init__(self, n_points, avg_acc=False, proc_noise_mult=0.003, meas_noise_mult=1):
    super(KalmanTracker, self).__init__()
    self.kalman = []
    self.n_points = n_points
    self.avg_acc = avg_acc
    self.dyn_params = 4
    self.meas_params = 2
    self.proc_noise_mult = proc_noise_mult
    self.meas_noise_mult = meas_noise_mult
    for _ in range(n_points):
      kalman_point = cv.KalmanFilter(dynamParams=self.dyn_params, measureParams=self.meas_params)
      kalman_point.measurementMatrix = np.array([[1,0,0,0],
                                                 [0,1,0,0]],
                                                np.float32)
      kalman_point.transitionMatrix = np.array([[1,0,1,0],
                                                [0,1,0,1],
                                                [0,0,1,0],
                                                [0,0,0,1]],
                                              np.float32)
      kalman_point.processNoiseCov = np.array([[1,0,0,0],
                                               [0,1,0,0],
                                               [0,0,1,0],
                                               [0,0,0,1]],
                                              np.float32) * self.proc_noise_mult
      kalman_point.measurementNoiseCov = np.array([[1,0],
                                                   [0,1]],
                                                  np.float32) * self.meas_noise_mult
      self.kalman.append(kalman_point)

  def predict(self, gt_points=None):
    pred_corners = []
    if gt_points is not None:
      for point, kalman in zip(gt_points, self.kalman):
        curr_pred = self._update_point(kalman, point[0], point[1])
        pred_corners.append(Point(curr_pred[0][0], curr_pred[1][0]).to_tuple())
    else:
      for kalman in self.kalman:
        curr_pred = self._update_point(kalman)
        pred_corners.append(Point(curr_pred[0][0], curr_pred[1][0]).to_tuple())
    return np.array(pred_corners, dtype=np.float32)

  def _update_point(self, kalman, x=None, y=None):
    if x is not None and y is not None:
      curr_mes = np.array([[np.float32(x)],[np.float32(y)]])
      kalman.correct(curr_mes)
    curr_pred = kalman.predict()
    return curr_pred

  def reset(self, points):
    n = min(self.n_points,len(points))
    if self.avg_acc:
      avg_acc0 = sum([ self.kalman[i].statePost[2] for i in range(n)])/n
      avg_acc1 = sum([ self.kalman[i].statePost[3] for i in range(n)])/n
    for i in range(n):
      kalman_point = self.kalman[i]
      if self.avg_acc:
        state = np.array([[points[i][0]],[points[i][1]],[avg_acc0],[avg_acc1]], dtype=np.float32)
      else:
        state = np.array([[points[i][0]],[points[i][1]],[0],[0]], dtype=np.float32)
      kalman_point.statePre = state
      kalman_point.statePost = state

