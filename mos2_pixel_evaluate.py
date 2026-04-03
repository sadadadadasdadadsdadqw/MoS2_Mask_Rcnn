import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from skimage import io
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import json
from datetime import datetime
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# Project root directory
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

# Import Mask RCNN related modules
from mrcnn import utils
import mrcnn.model as modellib

# Log and model directory
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_MoS2_400.h5")

# Configuration parameters
import mos2_train


class InferenceConfig(mos2_train.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3  # 1 background class + 3 target classes
    DETECTION_MIN_CONFIDENCE = 0.7


config = InferenceConfig()


class COCOMaskEvaluator:
    """COCO format based mask evaluator (Simplified: calculates overall pixel-level metrics only)"""

    def __init__(self, model, coco_gt_path):
        self.model = model
        self.coco_gt = COCO(coco_gt_path)

    def load_gt_masks_for_image(self, image_id):
        """Load the ground truth binary mask for a specified image"""
        ann_ids = self.coco_gt.getAnnIds(imgIds=image_id)
        annotations = self.coco_gt.loadAnns(ann_ids)

        img_info = self.coco_gt.loadImgs(image_id)[0]
        height, width = img_info['height'], img_info['width']

        class_mask = np.zeros((height, width), dtype=np.uint8)

        for ann in annotations:
            rle = ann['segmentation']
            if isinstance(rle, list):
                mask = self.coco_gt.annToMask(ann)
            else:
                mask = maskUtils.decode(rle)
            class_mask[mask > 0] = 255

        binary_mask = (class_mask > 0).astype(np.uint8) * 255

        return {
            'binary_mask': binary_mask,
            'image_info': img_info
        }

    def generate_prediction_masks(self, image):
        """Generate predicted binary mask"""
        results = self.model.detect([image], verbose=0)
        r = results[0]

        height, width = image.shape[:2]
        pred_binary_mask = np.zeros((height, width), dtype=np.uint8)

        if r['masks'].size > 0:
            combined = np.any(r['masks'], axis=-1).astype(np.uint8)
            pred_binary_mask = combined * 255

        return {
            'binary_mask': pred_binary_mask,
        }

    def calculate_mask_metrics(self, pred_mask, gt_mask):
        """Calculate overall pixel-level metrics"""
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

        pred_binary = (pred_mask > 0).astype(np.uint8).flatten()
        gt_binary = (gt_mask > 0).astype(np.uint8).flatten()

        precision = precision_score(gt_binary, pred_binary, zero_division=0)
        recall = recall_score(gt_binary, pred_binary, zero_division=0)
        f1 = f1_score(gt_binary, pred_binary, zero_division=0)
        iou = jaccard_score(gt_binary, pred_binary, zero_division=0)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou)
        }

    def evaluate_single_image(self, image_path, image_id):
        """Evaluate a single image"""
        try:
            image = io.imread(image_path)
            if image is None:
                return None

            gt_data = self.load_gt_masks_for_image(image_id)
            pred_data = self.generate_prediction_masks(image)

            pixel_metrics = self.calculate_mask_metrics(
                pred_data['binary_mask'], gt_data['binary_mask']
            )

            return {
                'image_id': image_id,
                'image_path': image_path,
                'pixel_metrics': pixel_metrics
            }

        except Exception as e:
            print(f"Error evaluating image {image_path}: {str(e)}")
            return None

    def evaluate_dataset(self, image_dir, output_dir=None, max_images=None):
        """Evaluate the entire dataset"""
        image_ids = self.coco_gt.getImgIds()

        if max_images:
            image_ids = image_ids[:max_images]

        print(f"Starting evaluation of overall pixel-level metrics for {len(image_ids)} images...")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results = []
        for i, image_id in enumerate(image_ids):
            img_info = self.coco_gt.loadImgs(image_id)[0]
            filename = img_info['file_name']
            image_path = os.path.join(image_dir, filename)

            if not os.path.exists(image_path):
                continue

            result = self.evaluate_single_image(image_path, image_id)
            if result:
                results.append(result)

            print(f"Progress: {i + 1}/{len(image_ids)} - Image ID: {image_id}")

        summary = self.calculate_summary_metrics(results)

        if output_dir:
            self.save_results(summary, output_dir)

        return summary, results

    def calculate_summary_metrics(self, results):
        """Calculate summary pixel-level metrics"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images_evaluated': len(results),
            'pixel_metrics': {}
        }

        if results:
            pixel_metrics_list = [r['pixel_metrics'] for r in results]
            for metric in ['precision', 'recall', 'f1_score', 'iou']:
                values = [m[metric] for m in pixel_metrics_list]
                summary['pixel_metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }

        return summary

    def save_results(self, summary, output_dir):
        """Save evaluation results to TXT and JSON"""
        # Save JSON summary
        with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Generate simplified TXT report
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(
                "Mask RCNN Overall Pixel-Level Evaluation Report (Tailored for Materials Science Area Evaluation)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Evaluation Time: {summary['timestamp']}\n")
            f.write(f"Total Images Evaluated: {summary['total_images_evaluated']}\n\n")
            f.write("Overall Pixel-Level Metrics (Mean ± Std):\n")
            f.write("-" * 40 + "\n")

            for metric, stats in summary['pixel_metrics'].items():
                f.write(f"{metric.upper():12}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")

        print(f"Evaluation report saved to: {report_path}")


def load_model():
    """Load model"""
    device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"
    with tf.device(device):
        model = modellib.MaskRCNN(
            mode="inference",
            model_dir=MODEL_DIR,
            config=config
        )

    if not os.path.exists(COCO_MODEL_PATH):
        raise FileNotFoundError(f"Model weights file does not exist: {COCO_MODEL_PATH}")

    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model


if __name__ == "__main__":
    # Path configuration
    IMAGE_DIR = os.path.join(ROOT_DIR, "MoS2/val2025")  # Test image directory
    COCO_GT_PATH = os.path.join(ROOT_DIR, "MoS2/annotations/instances_val2025.json")  # COCO format annotation file
    OUTPUT_DIR = os.path.join(ROOT_DIR, "evaluation_results")  # Result output directory

    if not os.path.exists(IMAGE_DIR):
        raise NotADirectoryError(f"Image directory does not exist: {IMAGE_DIR}")
    if not os.path.exists(COCO_GT_PATH):
        raise FileNotFoundError(f"COCO annotation file does not exist: {COCO_GT_PATH}")

    print("Loading model...")
    model = load_model()

    evaluator = COCOMaskEvaluator(model, COCO_GT_PATH)

    # max_images = None  # Evaluate all images
    summary, _ = evaluator.evaluate_dataset(IMAGE_DIR, OUTPUT_DIR, max_images=None)

    # Output only these 4 core metrics in the terminal
    print("\n" + "=" * 60)
    print("Evaluation completed! Overall pixel-level metrics are as follows:")
    print(f"Total Images Evaluated: {summary['total_images_evaluated']}")
    print("-" * 60)
    print(f"Pixel Precision : {summary['pixel_metrics']['precision']['mean']:.4f}")
    print(f"Pixel Recall    : {summary['pixel_metrics']['recall']['mean']:.4f}")
    print(f"Pixel F1-Score  : {summary['pixel_metrics']['f1_score']['mean']:.4f}")
    print(f"Pixel IoU       : {summary['pixel_metrics']['iou']['mean']:.4f}")
    print("=" * 60)