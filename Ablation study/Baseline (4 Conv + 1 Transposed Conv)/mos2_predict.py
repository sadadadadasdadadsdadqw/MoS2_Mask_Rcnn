import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage import io
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_Baseline_400.h5")

# Configuration
import mos2_train
config = mos2_train.CocoConfig()
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3
    DETECTION_MIN_CONFIDENCE = 0.7
config = InferenceConfig()

# Create model

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Prediction and saving function
def predict_and_save(input_dir, output_dir, mask_dir,mask2_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mask2_dir, exist_ok=True)
    valid_extensions = ['.jpg', '.jpeg', '.png']

    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"result_{filename}")
            mask_path = os.path.join(mask_dir, f"mask_{os.path.splitext(filename)[0]}.png")
            mask2_path = os.path.join(mask2_dir, f"mask_{os.path.splitext(filename)[0]}.png")
            image = io.imread(image_path)
            orig_shape = image.shape
            results = model.detect([image], verbose=0)
            r = results[0]
            result_image, binary_mask,mask2 = visualize.display_instances(
                image, r['rois'], r['masks'], r['class_ids'],
                ['', 'single layer', 'contaminants', 'multi layer'], r['scores']
            )
            result_image = cv2.resize(result_image, (orig_shape[1], orig_shape[0]))
            cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            binary_mask = cv2.resize(binary_mask, (orig_shape[1], orig_shape[0]))
            cv2.imwrite(mask_path, binary_mask)
            binary2_mask = cv2.resize(mask2, (orig_shape[1], orig_shape[0]))
            cv2.imwrite(mask2_path, binary2_mask)

            print(f"Saved: {output_path} and {mask_path}")

# Path configuration
INPUT_FOLDER = os.path.join(ROOT_DIR, "Bimage")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "result")
MASK_FOLDER = os.path.join(ROOT_DIR, "masks")
MASK_FOLDER2 = os.path.join(ROOT_DIR, "masks2")

# Run prediction
predict_and_save(INPUT_FOLDER, OUTPUT_FOLDER, MASK_FOLDER, MASK_FOLDER2)
