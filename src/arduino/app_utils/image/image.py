# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import io
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
from arduino.app_utils import Logger

logger = Logger(__name__)


from enum import Enum

class Shape(str, Enum):
    RECTANGLE = "rectangle"
    CIRCLE = "circle"



# Define a mapping of confidence ranges to colors for bounding boxes (hex and precomputed text color for contrast)
CONFIDENCE_MAP = {
    (0, 20):   {"bb": "#FF0976", "text": (255, 255, 255)},   # Pink, white text
    (21, 40):  {"bb": "#FF8131", "text": (0, 0, 0)},         # Orange, black text
    (41, 60):  {"bb": "#FFFC00", "text": (0, 0, 0)},         # Yellow, black text
    (61, 80):  {"bb": "#00DED7", "text": (0, 0, 0)},         # Light blue, black text
    (81, 100): {"bb": "#1EFF00", "text": (0, 0, 0)},         # Green, black text
}

FONT_PATH = "/home/app/.fonts/OpenSans.ttf"


# Get the color dict for a given confidence value based on the defined ranges.
# If the confidence is outside the defined ranges, it defaults to green.
def get_box_color(confid):
    for (low, high), color in CONFIDENCE_MAP.items():
        if low <= confid <= high:
            return color
    return ("#1EFF00", (0, 0, 0))  # If out of range, default to green, black text


def _read(file_path: str) -> bytes | None:
    """Read an image from a file path and return a PIL Image object."""
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return None


def get_image_type(image_bytes: bytes | Image.Image) -> str | None:
    """Detect the type of image from bytes or a PIL Image object.

    Returns:
        str: The image type in lowercase (e.g., 'jpeg', 'png').
        None if the image type cannot be determined.
    """
    try:
        if isinstance(image_bytes, Image.Image):
            # If the input is already a PIL Image, we can directly get its format
            if image_bytes.format is not None:
                return image_bytes.format.lower()
        elif isinstance(image_bytes, (bytes, bytearray, memoryview)):
            image = Image.open(io.BytesIO(image_bytes))
            return image.format.lower() if image.format is not None else None  # Returns 'jpeg', 'png', etc.
        return None
    except Exception as e:
        print(f"Error detecting image type: {e}")
        return None


def get_image_bytes(image: str | Image.Image | bytes) -> bytes | None:
    """Convert different type of image objects to bytes."""
    if image is None:
        return None
    try:
        if isinstance(image, Image.Image):
            byte_io = io.BytesIO()
            image.save(byte_io, "PNG")
            return byte_io.getvalue()
        elif isinstance(image, bytes):
            return image
        elif isinstance(image, str):
            return _read(image)
    except Exception as e:
        logger.error(f"Error converting image to bytes: {e}")
        return None


def draw_bounding_boxes(
    image: Image.Image | bytes,
    detection: dict,
    shape: Shape = Shape.RECTANGLE,
) -> Image.Image:
    """Draw bounding boxes on an image using PIL.

    The thickness of the box and font size are scaled based on image size.

    Args:
        image (Image.Image | bytes): The image to draw on, can be a PIL Image or bytes.
        detection (dict): A dictionary containing detection results with keys
            'class_name', 'bounding_box_xyxy', and 'confidence'.
        draw (ImageDraw.ImageDraw, optional): An existing ImageDraw object to use.
            If None, a new one is created.
        shape (Shape, optional): Shape of the bounding box. Defaults to rectangle.

    Returns:
        Image.Image: The annotated image with bounding boxes drawn.

    Raises:
        ValueError: If an unsupported shape is provided.
    """
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = Image.open(io.BytesIO(image))

    if not detection or "detection" not in detection:
        return image

    if shape not in (Shape.RECTANGLE, Shape.CIRCLE):
        raise ValueError(f"Unsupported shape '{shape}'.")

    draw = ImageDraw.Draw(image)

    detection = detection["detection"]

    # Scale font size and box thickness based on image size and number of detections
    max_dim = max(image.size)
    n_detections = max(1, len(detection))
    font_size = max(8, int(max_dim / (28 + n_detections * 3)))
    box_thickness = max(1, int(max_dim / 250))
    label_vpad = max(2, int(font_size * 0.4))
    label_hpad = max(4, int(font_size * 0.8))

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        logger.warning(f"Error loading custom font: {e}. Using default font.")
        font = ImageFont.load_default(14)

    for _, obj_det in enumerate(detection):
        if "class_name" not in obj_det or "bounding_box_xyxy" not in obj_det or "confidence" not in obj_det:
            continue

        class_name = obj_det["class_name"]
        box = obj_det["bounding_box_xyxy"]
        confidence = float(obj_det["confidence"])
        x1, y1, x2, y2 = map(int, box[:4])

        box_color, text_color = get_box_color(confidence)

        # Draw the bounding box
        if shape == Shape.CIRCLE:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            radius = 10
            bounding_box = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
            draw.ellipse(bounding_box, outline=box_color, width=2)
        else:
            draw.rectangle((x1, y1, x2, y2), outline=box_color, width=box_thickness)

        # Prepare the label
        text = f"{class_name.capitalize()} {confidence:.1f}%"
        left, top, right, bottom = font.getbbox(text)
        text_width, text_height = right - left, bottom - top
        label_gap = max(1, int(font_size * 0.15))
        y1_text = y1 - text_height - label_vpad * 2 - label_gap  # Above the box
        if y1_text < 0:
            y1_text = y1 + label_gap  # Below the box if not enough space above
        y2_text = y1_text + text_height + label_vpad * 2
        x2_text = x1 + text_width + label_hpad * 2

        # Draw the label background and text
        draw.rectangle((x1, y1_text, x2_text, y2_text), fill=box_color, outline=None)
        draw.text((x1 + label_hpad, y1_text + label_vpad), text, fill=text_color, font=font)

    return image


def draw_anomaly_markers(image: Image.Image | bytes, detection: dict, draw: ImageDraw.ImageDraw | None = None) -> Image.Image | None:
    """Draw bounding boxes on an image using PIL.

    The thickness of the box and font size are scaled based on image size.

    Args:
        image (Image.Image | bytes): The image to draw on, can be a PIL Image or bytes.
        detection (dict): A dictionary containing detection results with keys 'class_name', 'bounding_box_xyxy', and
            'score'.
        draw (ImageDraw.ImageDraw, optional): An existing ImageDraw object to use. If None, a new one is created.
    """
    if isinstance(image, (bytes, bytearray, memoryview)):
        image_box = Image.open(io.BytesIO(image))
    else:
        image_box = image

    if image_box.mode != "RGBA":
        image_box = image_box.convert("RGBA")

    if draw is None:
        draw = ImageDraw.Draw(image_box)

    max_anomaly_score = detection.get("anomaly_max_score", 0.0)

    if not detection or "detection" not in detection:
        return None
    detection = detection["detection"]

    # Scale font size and box thickness based on image size
    ref_dim = max(image_box.size)
    box_thickness = max(1, int(ref_dim / 400))

    for i, obj_det in enumerate(detection):
        if "class_name" not in obj_det or "bounding_box_xyxy" not in obj_det or "score" not in obj_det:
            continue

        box = obj_det["bounding_box_xyxy"]
        score = float(obj_det["score"])

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        normalized_score = score / max_anomaly_score if max_anomaly_score > 0 else 0
        alpha = int(255 * min(max(normalized_score, 0), 1))

        base_color_rgb = (255, 0, 0)
        outline_color = (0, 0, 0)
        fill_color_with_alpha = base_color_rgb + (alpha,)

        temp_layer = Image.new("RGBA", image_box.size, (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_layer)

        temp_draw.rectangle([x1, y1, x2, y2], fill=fill_color_with_alpha)
        temp_draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=box_thickness)
        image_box = Image.alpha_composite(image_box, temp_layer)

        draw = ImageDraw.Draw(image_box)

    return image_box
