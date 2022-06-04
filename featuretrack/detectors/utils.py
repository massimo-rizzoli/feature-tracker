from typing import Tuple

class Point:
  def __init__(self, x=-1.0, y=-1.0):
    self.x = x
    self.y = y

  def __str__(self):
    return f'({self.x}, {self.y})'

  def __repr__(self):
    return self.__str__()

  def to_tuple(self) -> Tuple[float,float]:
    return (self.x,self.y)

  @classmethod
  def from_GFT(cls, gft_point) -> 'Point':
    x, y = gft_point.ravel()
    return Point(x,y)

  @classmethod
  def from_SIFT(cls, sift_point) -> 'Point':
    x, y = sift_point.pt
    return Point(x,y)
