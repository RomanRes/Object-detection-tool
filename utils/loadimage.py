from idlelib.pathbrowser import PathBrowser

import numpy as np
import base64
import io

from PIL import Image
from tensorflow.keras.utils import img_to_array
from pathlib import Path

DEFAULT_IMAGE = Path("img/000000000057.jpg")


def load_image_pixels(contents):
    print("start load_image_pixels")

    if contents:
        image = contents.split(",")[1]
        image = base64.b64decode(image)
        image = Image.open(io.BytesIO(image)).convert("RGB")
    else:
        # Default image if nothing uploaded
        if not DEFAULT_IMAGE.exists():
            raise FileNotFoundError("Default image not found in assets/")
        image = Image.open(DEFAULT_IMAGE).convert("RGB")

    return image


def resize_and_scale(image, shape):
    # resizing to network input size (416, 416)
    # and converting to ndarray
    image = image.resize(shape)
    image = img_to_array(image)
    print(image.shape, "numpy shape")

    # scaling
    image = image.astype("float16")
    image /= 255.0

    # add one dimension to make a size on one batch
    image = np.expand_dims(image, 0)

    return image
