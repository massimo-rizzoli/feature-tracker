from cv2 import cv2 as cv


class VideoWriter:

  def __init__(self, filename, fps, resolution):
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    self.writer = cv.VideoWriter(''.join([filename,'.avi']), fourcc, fps, resolution)

  def write(self, frame):
    self.writer.write(frame)

  def release(self):
    self.writer.release()
