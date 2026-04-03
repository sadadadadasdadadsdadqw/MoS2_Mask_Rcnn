"""
Mask R-CNN
OpenCV-based Visualization Functions.
"""

import cv2
import numpy as np
import colorsys
import random


def random_colors(N, bright=True):
    """Generate random colors."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply mask to image using OpenCV."""
    mask = mask.astype(bool)
    for c in range(3):
        image[:, :, c] = np.where(mask,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image.astype(np.uint8)


def draw_border(image, box, color, thickness=2):
    """Draw bounding box with OpenCV."""
    y1, x1, y2, x2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def display_instances(image, boxes, masks, class_ids, class_names, scores=None,
                      show_mask=True, show_bbox=True, colors=None):
    """
    OpenCV implementation of instance visualization.
    Returns:
        visualized_image: Visualized image (RGB format)
        all_mask: Complete mask containing all detected instances (binary/black-and-white image)
        class2_mask: Dedicated mask containing only class_id=2 (binary/black-and-white image)
    """
    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    orig_image = image.copy()

    N = boxes.shape[0]

    all_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    class2_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if N == 0:

        return cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB), all_mask, class2_mask

    fixed_colors = {
        1: (0, 0, 0),
        2: (0, 155, 155),
        3: (155, 155, 0)
    }

    if colors is None:
        colors = []
        for i in range(N):
            class_id = class_ids[i]
            if class_id in fixed_colors:
                colors.append(fixed_colors[class_id])
            else:
                random.seed(class_id)
                colors.append([int(c * 255) for c in colorsys.hsv_to_rgb(
                    random.random(), 1.0, 1.0)][::-1])  # RGB->BGR


    for i in range(N):
        color = colors[i]
        class_id = class_ids[i]
        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]


        all_mask = np.where(mask, 255, all_mask)

        if class_id == 3:
            class2_mask = np.where(mask, 255, class2_mask)


        if show_bbox:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


        if show_mask:
            mask_bool = mask.astype(bool)
            alpha = 0.6
            for c in range(3):
                image[:, :, c] = np.where(
                    mask_bool,
                    image[:, :, c] * (1 - alpha) + alpha * color[c],
                    image[:, :, c]
                )


        label = class_names[class_id]
        score = scores[i] if scores is not None else None
        text = f"{label} {score:.2f}" if score else label

        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1
        )
        y_text = max(y1 - 10, text_height + 5)


        cv2.rectangle(image, (x1, y1 - text_height - 10),
                      (x1 + text_width, y1), color, -1)

        cv2.putText(image, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


    final_image = cv2.addWeighted(orig_image, 0.4, image, 0.6, 0)
    return cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB), all_mask, class2_mask