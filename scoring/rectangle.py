"""
Copyright 2018 Defense Innovation Unit Experimental All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


class Rectangle(object):
    """Rectangle class."""

    def __init__(self, xmin, ymin, xmax, ymax):
        """Constructs a Rectangle instance."""
        if xmin >= xmax or ymin >= ymax:
            self.xmin_ = None
            self.ymin_ = None
            self.xmax_ = None
            self.ymax_ = None
        else:
            self.xmin_ = xmin
            self.ymin_ = ymin
            self.xmax_ = xmax
            self.ymax_ = ymax
        self.coords = (xmin, ymin, xmax, ymax)

    def __eq__(self, other):
        """== operator overloading."""
        return (
            (self.xmin_ == other.xmin_)
            and (self.ymin_ == other.ymin_)
            and (self.xmax_ == other.xmax_)
            and (self.ymax_ == other.ymax_)
        )

    def __ne__(self, other):
        """!= operator overloading."""
        return not self.__eq__(other)

    def is_empty(self):
        """Determines if the Rectangle instance is valid or not."""
        return (
            (self.xmin_ is None)
            or (self.ymin_ is None)
            or (self.xmax_ is None)
            or (self.ymax_ is None)
            or (self.xmin_ >= self.xmax_)
            or (self.ymin_ >= self.ymax_)
        )

    def width(self):
        """Returns the width of the Rectangle instance."""
        return self.xmax_ - self.xmin_

    def height(self):
        """Returns the height of the Rectangle instance."""
        return self.ymax_ - self.ymin_

    def area(self):
        """Returns the area of the Rectangle instance."""
        return self.width() * self.height()

    def intersect(self, other):
        """Returns the intersection of this rectangle with the other rectangle."""

        xmin = max(self.xmin_, other.xmin_)
        ymin = max(self.ymin_, other.ymin_)
        xmax = min(self.xmax_, other.xmax_)
        ymax = min(self.ymax_, other.ymax_)
        return Rectangle(xmin, ymin, xmax, ymax)

    def intersects(self, other):
        """Tests if this rectangle has an intersection with another rectangle."""
        return not (
            self.is_empty()
            or other.is_empty()
            or (other.xmax_ <= self.xmin_)
            or (self.xmax_ <= other.xmin_)
            or (other.ymax_ <= self.ymin_)
            or (self.ymax_ <= other.ymin_)
        )

    def contains(self, x, y):
        """Tests if a point is inside or on any of the edges of the rectangle."""
        return (x >= self.xmin_) and (x <= self.xmax_) and (y >= self.ymin_) and (y <= self.ymax_)

    def intersect_over_union(self, other):
        """Returns the intersection over union ratio of this and other rectangle."""
        if not self.intersects(other):
            return 0.0

        intersect_rect = self.intersect(other)
        if intersect_rect.is_empty():
            return 0.0

        if self.area() == 0 or other.area() == 0:
            return 0.0

        return intersect_rect.area() / (self.area() + other.area() - intersect_rect.area())
