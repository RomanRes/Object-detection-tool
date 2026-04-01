import numpy as np
from typing import List, Tuple, Optional, Any, Union

# Local imports
from utils.classbox import BoundBox
from utils.loadimage import resize_and_scale
from detection.model import get_model


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the sigmoid activation function.

    Args:
        x: Input value or numpy array.

    Returns:
        The sigmoid of x.
    """
    return 1.0 / (1.0 + np.exp(-x))


def decode_netout(
        netout: np.ndarray,
        anchors: List[float],
        obj_thresh: float,
        img_h: int,
        img_w: int,
        nb_box: int = 3
) -> np.ndarray:
    """
    Decodes the raw network output for a specific grid size into bounding boxes.

    Args:
        netout: Raw output array from the network.
        anchors: Predefined anchor box dimensions.
        obj_thresh: Confidence threshold for object detection.
        img_h: Target image height.
        img_w: Target image width.
        nb_box: Number of anchor boxes per grid cell.

    Returns:
        np.ndarray: Filtered boxes with coordinates and class probabilities.
    """
    grid_h, grid_w = netout.shape[:2]
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    # Apply sigmoid to x, y, objectness and classes
    netout[..., :2] = sigmoid(netout[..., :2])
    netout[..., 4:] = sigmoid(netout[..., 4:])

    # Calculate final class probabilities (objectness * class_scores)
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    # Create grid offsets for vectorized coordinate calculation
    _columns_array = np.arange(grid_h).repeat(grid_h * nb_box).reshape((grid_h, grid_w, nb_box, 1))
    _rows_array = np.arange(grid_w)
    _rows_array = np.repeat(_rows_array, nb_box, axis=0)
    _rows_array = np.tile(_rows_array, (1, grid_w)).reshape((grid_h, grid_w, nb_box, 1))

    # Convert x, y relative to grid and normalize to image size
    y_arr = (netout[..., 1:2] + _columns_array) / grid_h
    x_arr = (netout[..., 0:1] + _rows_array) / grid_w

    # Convert w, h using anchors and normalize to image size
    w_and_h = np.exp(netout[..., 2:4]) * np.array(anchors).reshape((nb_box, 2)) / img_w

    # Concatenate results and filter by threshold
    boxes = np.concatenate((x_arr, y_arr, w_and_h, netout[..., 4:]), axis=3)
    boxes = boxes[(boxes[..., 4:5] > obj_thresh).all(axis=3)]

    return boxes


def convert_coordinates_to_minmax(boxes: np.ndarray) -> np.ndarray:
    """
    Converts box coordinates from (x, y, w, h) center format to (xmin, ymin, xmax, ymax).

    Args:
        boxes: Numpy array of boxes in (x, y, w, h, ...) format.

    Returns:
        np.ndarray: Boxes in min/max format.
    """
    x = boxes[..., 0:1]
    y = boxes[..., 1:2]
    w = boxes[..., 2:3]
    h = boxes[..., 3:4]

    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2

    return np.concatenate((x_min, y_min, x_max, y_max, boxes[..., 4:]), axis=1)


def resize_to_bild_size(boxes: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Scales relative coordinates (0-1) to absolute pixel values based on image size.

    Args:
        boxes: Array of boxes with relative coordinates.
        image_size: Tuple of (width, height) of the target image.

    Returns:
        np.ndarray: Boxes with absolute pixel coordinates.
    """
    width, height = image_size
    boxes[..., 0] *= width
    boxes[..., 1] *= height
    boxes[..., 2] *= width
    boxes[..., 3] *= height

    boxes[..., 0:4] = boxes[..., 0:4].astype(int)
    return boxes


def bbox_iou(box1: BoundBox, box2: BoundBox) -> float:
    """
    Calculates Intersection over Union (IoU) between two BoundBox objects.
    """
    x1 = max(box1.xmin, box2.xmin)
    x2 = min(box1.xmax, box2.xmax)
    y1 = max(box1.ymin, box2.ymin)
    y2 = min(box1.ymax, box2.ymax)

    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    if intersect == 0:
        return 0.0

    box1Area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    box2Area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)

    return intersect / float(box1Area + box2Area - intersect)


def do_nms(boxes: np.ndarray, nms_thresh: float) -> List[np.ndarray]:
    """
    Performs Non-Maximum Suppression (NMS) to eliminate redundant overlapping boxes.

    Args:
        boxes: Array of all detected boxes.
        nms_thresh: IoU threshold for suppression.

    Returns:
        List[np.ndarray]: List of unique bounding boxes.
    """
    if len(boxes) == 0:
        return []

    clear_boxes = []
    # Identify unique classes present in the detected boxes (starting from index 5)
    exist_classes = np.where(np.any(boxes[..., 5:] != 0, axis=0) == True)[0] + 5

    for c in exist_classes:
        # Filter boxes for current class
        selected_boxes = boxes[(boxes[:, c] > 0)]

        # Precompute areas for overlap calculation
        areas = (selected_boxes[:, 2] - selected_boxes[:, 0] + 1) * \
                (selected_boxes[:, 3] - selected_boxes[:, 1] + 1)

        # Sort boxes by confidence score (descending)
        sort_indexes = np.flip(np.argsort(selected_boxes[:, 4]))

        while sort_indexes.size != 0:
            current_box_index = sort_indexes[0]
            clear_boxes.append(selected_boxes[current_box_index])
            sort_indexes = sort_indexes[1:]

            if sort_indexes.size == 0:
                break

            # Calculate intersection coordinates
            xx1 = np.maximum(selected_boxes[current_box_index][0], selected_boxes[sort_indexes, 0])
            yy1 = np.maximum(selected_boxes[current_box_index][1], selected_boxes[sort_indexes, 1])
            xx2 = np.minimum(selected_boxes[current_box_index][2], selected_boxes[sort_indexes, 2])
            yy2 = np.minimum(selected_boxes[current_box_index][3], selected_boxes[sort_indexes, 3])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Ratio of overlap
            overlap = (w * h) / areas[sort_indexes]
            # Delete boxes with overlap higher than threshold
            sort_indexes = np.delete(sort_indexes, np.argwhere(overlap > nms_thresh))

    return clear_boxes


def convert_to_BoundBox_class(boxes: List[np.ndarray]) -> List[BoundBox]:
    """Converts raw arrays into a list of BoundBox objects."""
    converted_boxes = []
    for _ in boxes:
        box = BoundBox(_[0], _[1], _[2], _[3], _[4], _[5:])
        converted_boxes.append(box)
    return converted_boxes


# -----------------------------------------------------------------------------
# MODEL SINGLETON
# -----------------------------------------------------------------------------
_model: Optional[Any] = None


def get_loaded_model() -> Any:
    """Ensures the model is only loaded once (Singleton pattern)."""
    global _model
    if _model is None:
        _model = get_model()
    return _model


# Initialize the model once
model = get_loaded_model()


def predict_boxes(
        image: Any,
        class_threshold: float,
        nms_thresh: float,
        ANCHORS: List[List[float]],
        IMG_SIZE: Tuple[int, int]
) -> List[BoundBox]:
    """
    Full pipeline: Preprocess image, predict with YOLO, decode, NMS, and return boxes.

    Args:
        image: PIL Image object.
        class_threshold: Threshold for filtering low-confidence objects.
        nms_thresh: Threshold for Non-Maximum Suppression.
        ANCHORS: YOLO anchor configuration.
        IMG_SIZE: Input size for the neural network.

    Returns:
        List[BoundBox]: A list of detected objects.
    """
    input_w, input_h = IMG_SIZE
    image_w, image_h = image.size

    # Prepare image for the network
    processed_image = resize_and_scale(image, (input_w, input_h))

    # Inference
    yhat = model.predict(processed_image)

    # Post-processing
    boxes = np.empty([0, 85])
    for i in range(len(yhat)):
        # Decode grid outputs for each scale
        b = decode_netout(yhat[i][0], ANCHORS[i], class_threshold, input_h, input_w)
        if b.shape[0] > 0:
            boxes = np.concatenate([boxes, b])

    if len(boxes) > 0:
        # Transform coordinates
        boxes = convert_coordinates_to_minmax(boxes)
        boxes = resize_to_bild_size(boxes, (image_w, image_h))
        # Filter overlapping detections
        boxes = do_nms(boxes, nms_thresh)
        # Wrap results in professional objects
        boxes = convert_to_BoundBox_class(boxes)
    else:
        boxes = []

    return boxes