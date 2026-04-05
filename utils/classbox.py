import numpy as np
from typing import List, Optional, Union, Any
from parameters.parameters import LABELS


class BoundBox:
    """
    Data container for a single YOLOv3 detection.
    It stores spatial coordinates and computes the most likely class label.
    """

    def __init__(
            self,
            xmin: Union[int, float],
            ymin: Union[int, float],
            xmax: Union[int, float],
            ymax: Union[int, float],
            objness: Optional[float] = None,
            classes: Any = None
    ) -> None:
        """
        Initializes a bounding box with coordinates and classification data.

        Args:
            xmin, ymin, xmax, ymax: Coordinates of the box.
            objness: Confidence score that an object exists.
            classes: Numpy array containing probabilities for each class.
        """
        self.labels: List[str] = LABELS
        self.xmin: Union[int, float] = xmin
        self.ymin: Union[int, float] = ymin
        self.xmax: Union[int, float] = xmax
        self.ymax: Union[int, float] = ymax
        self.objness: Optional[float] = objness
        self.classes: Any = classes

        # Determine label and initial score
        if classes is not None and len(classes) > 0:
            # np.argmax finds the index of the highest probability
            self.label: Optional[str] = self.labels[np.argmax(classes)]
            self.score: float = float(np.max(classes))
        else:
            self.label = None
            self.score = 0.0

        # Final score assignment from the class probability array
        self.score = np.max(self.classes)
