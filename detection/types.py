from dataclasses import dataclass
from typing import Optional, Dict

from detection.pandaset.util import LabelClass
from detection.pandaset.dataset import DETECTING_CLASSES

import torch


@dataclass
class Detections:
    """Dataclass for 2D bounding box detections.

    Args:
        centroids: Dict[LabelClass, [N x 2]] centroids tensor. Each row is (x, y).
        yaws: Dict[LabelClass, [N]] rotations in radians tensor.
        boxes: Dict[LabelClass, [N x 2]] boxes tensor. Each row is (x_size, y_size).
        scores: Dict[LabelClass, [N]] detection scores tensor. None if ground truth.
    """

    centroids: Dict[LabelClass, torch.Tensor]
    yaws: Dict[LabelClass, torch.Tensor]
    boxes: Dict[LabelClass, torch.Tensor]
    scores: Dict[LabelClass, Optional[torch.Tensor]] = {class_: None for class_ in DETECTING_CLASSES}

    @property
    def centroids_x(self, class_: LabelClass) -> torch.Tensor:
        """Return the x-axis centroid coordinates."""
        return self.centroids[class_][:, 0]

    @property
    def centroids_y(self, class_: LabelClass) -> torch.Tensor:
        """Return the y-axis centroid coordinates."""
        return self.centroids[class_][:, 1]

    @property
    def boxes_x(self, class_: LabelClass) -> torch.Tensor:
        """Return the x-axis bounding box size."""
        return self.boxes[class_][:, 0]

    @property
    def boxes_y(self, class_: LabelClass) -> torch.Tensor:
        """Return the y-axis bounding box size."""
        return self.boxes[class_][:, 1]

    def to(self, device: torch.device) -> "Detections":
        """Return a copy of the detections moved to another device."""
        return Detections(
            {self.centroids[class_].to(device) for class_ in DETECTING_CLASSES},
            {self.yaws[class_].to(device) for class_ in DETECTING_CLASSES},
            {self.boxes[class_].to(device) for class_ in DETECTING_CLASSES},
            {self.scores[class_].to(device) for class_ in DETECTING_CLASSES},
        )

    def __len__(self, class_: LabelClass) -> int:
        """Return the number of detections."""
        return len(self.centroids[class_])
