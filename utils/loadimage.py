import numpy as np
import base64
import io
from PIL import Image
from tensorflow.keras.utils import img_to_array
from pathlib import Path
from typing import Optional, Tuple, Any, Union

# Define the path for the placeholder image used on app startup
DEFAULT_IMAGE: Path = Path("img/000000000057.jpg")


def load_image_pixels(contents: Optional[str]) -> Image.Image:
    """
    Handles image loading from a Dash upload component or a local default file.

    Args:
        contents: Base64 encoded string from dcc.Upload or None.

    Returns:
        PIL.Image.Image: The loaded image in RGB format.

    Raises:
        FileNotFoundError: If the default image is missing and no content is provided.
    """
    print("🚀 Execution: load_image_pixels")

    if contents:
        # Decode the base64 string provided by Dash
        # Dash base64 strings look like: "data:image/jpeg;base64,/9j/4AAQ..."
        image_data: str = contents.split(",")[1]
        decoded_image: bytes = base64.b64decode(image_data)
        image: Image.Image = Image.open(io.BytesIO(decoded_image)).convert("RGB")
    else:
        # Fallback to local default image if no file was uploaded
        if not DEFAULT_IMAGE.exists():
            raise FileNotFoundError(f"Default image not found at: {DEFAULT_IMAGE}")
        image = Image.open(DEFAULT_IMAGE).convert("RGB")

    return image


def resize_and_scale(image: Image.Image, shape: Tuple[int, int]) -> np.ndarray:
    """
    Prepares the PIL image for the neural network (YOLOv3).

    Args:
        image: The original PIL image.
        shape: The required input shape for the network (e.g., 416, 416).

    Returns:
        np.ndarray: A 4D normalized numpy array (Batch, Height, Width, Channels).
    """
    # 1. Resize to network input size (usually 416x416)
    image_resized = image.resize(shape)

    # 2. Convert PIL Image object to a numerical numpy array
    image_array: np.ndarray = img_to_array(image_resized)
    print(f"📏 Numpy shape after array conversion: {image_array.shape}")

    # 3. Scaling / Normalization
    # Convert to float16 to save memory and scale pixels to [0, 1] range
    image_scaled: np.ndarray = image_array.astype("float16")
    image_scaled /= 255.0

    # 4. Expand dimensions
    # Keras models expect a batch dimension: (Batch_Size, H, W, C)
    # Becomes (1, 416, 416, 3)
    final_batch: np.ndarray = np.expand_dims(image_scaled, 0)

    return final_batch
