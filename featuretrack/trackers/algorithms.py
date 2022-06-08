import numpy as np
from cv2 import cv2 as cv
from featuretrack.trackers.trackers import LucasKanadeTracker, SIFTTracker, KalmanTracker


def lucas_kanade(args):
  cap = args.cap
  fe = args.fe
  tracker = LucasKanadeTracker()

  t = 0
  while cap.isOpened():
    ret, frame = cap.read()

    if cv.waitKey(args.delay) == ord('q') or not ret:
      cap.release()
      break

    frame = cv.resize(frame, args.resolution)
    frame_copy = frame.copy()

    if t % args.interval == 0:
      points = fe.get_feature_points(frame)
    else:
      points = tracker.predict(prev_frame, frame, prev_points)

    prev_frame = frame.copy()
    prev_points = points

    np_points = points.astype(int)
    step = 255/len(np_points)
    for i, point in enumerate(np_points):
      x, y = point
      colour = np.array([i*step % 255, 2*i*step % 255, 255-(i*step)], dtype=np.float64)
      cv.circle(frame_copy, (x,y), args.radious, colour, thickness=args.thickness)

    name = f'Lucas-Kanade'
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.imshow(name, frame_copy)

    if args.writer:
      args.writer.write(frame_copy)

    t += 1
  cv.destroyAllWindows()
  if args.writer:
    args.writer.release()


def siftbf(args):
  cap = args.cap
  fe = args.fe
  tracker = SIFTTracker(args.points, args.thresh, fe)
  img_display = None

  t = 0
  while cap.isOpened():
    ret, frame = cap.read()

    key = cv.waitKey(args.delay)
    if key == ord('q') or not ret:
      cap.release()
      break
    # Press space to pause to better see the matches
    if not args.hidematch and key == ord(' '):
      print('Press <space> to resume...')
      while cv.waitKey(1000) != ord(' '):
        pass

    frame = cv.resize(frame, args.resolution)
    frame_copy = frame.copy()

    if t % args.interval == 0:
      keypoints, desc = fe.get_features(frame)
      points = fe._keypoints_to_tuple_list(keypoints)
      prev_frame = frame.copy()
    else:
      points, curr_kp, curr_desc, good_matches = tracker.predict(prev_desc, frame)
      if not args.hidematch:
        img_match = cv.drawMatches(img1=prev_frame, keypoints1=keypoints, img2=frame,
                                   keypoints2=curr_kp, matches1to2=good_matches, outImg=None,
                                   flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
      np_points = points.astype(int)
      for point in np_points:
        x, y = point
        colour = np.array([0, 0, 200], dtype=np.float64)
        cv.circle(frame_copy, (x,y), args.radious, colour, thickness=args.thickness)

      if not args.hidematch:
        img_display = np.concatenate((img_match, frame_copy), axis=1)
      else:
        img_display = frame_copy
      name = f'SIFT'
      cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
      cv.imshow(name, img_display)

    if args.writer and t != 0:
      args.writer.write(img_display)

    prev_desc = desc
    t += 1

  cv.destroyAllWindows()
  if args.writer:
    args.writer.release()


def kalman(args):
  cap = args.cap
  fe = args.fe
  gt_tracker = LucasKanadeTracker()
  tracker = KalmanTracker(args.points, avg_acc=args.avgacc)

  t = 0
  while cap.isOpened():
    ret, frame = cap.read()

    if cv.waitKey(args.delay) == ord('q') or not ret:
      cap.release()
      break

    frame = cv.resize(frame, args.resolution)
    frame_copy = frame.copy()

    if t % args.interval == 0:
      points = fe.get_feature_points(frame)
      pred_points = points
      tracker.reset(points)
    else:
      points = gt_tracker.predict(prev_frame, frame, prev_points)
      gt_points = None
      if t % args.correct == 0:
        gt_points = points
      pred_points = tracker.predict(gt_points=gt_points)

    prev_frame = frame.copy()
    prev_points = points

    np_points = points.astype(int)
    np_pred_points = pred_points.astype(int)
    red = np.array([0,0,200], dtype=np.float64)
    green = np.array([0,200,0], dtype=np.float64)
    for point, pred_point in zip(np_points, np_pred_points):
      x, y = point
      pred_x, pred_y = pred_point
      cv.circle(frame_copy, (x,y), args.radious, red, thickness=args.thickness)
      cv.circle(frame_copy, (pred_x,pred_y), args.radious, green, thickness=args.thickness)

    name = f'Kalman Filter'
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.imshow(name, frame_copy)

    if args.writer:
      args.writer.write(frame_copy)

    t += 1

  cv.destroyAllWindows()
  if args.writer:
    args.writer.release()
