# Deep Learning Model Details for BF-Image Processing

This repository contains the code and configuration details for the MoS2 crystal microscope image processing project, utilizing a Cascade 2D Flake Search approach.

## Repository Links
* **Stage 1 Training Code (`mos2_train.py`):** [Link](https://github.com/sadadadadasdadadsdadqw/MoS2_Mask_Rcnn/blob/master/mos2_train.py)
* **Dataset (`MoS2`):** [Link](https://github.com/sadadadadasdadadsdadqw/MoS2_Mask_Rcnn/tree/master/MoS2)
* **COCO Format Conversion Code (`json_to_coco.py.py`):** [Link](https://github.com/sadadadadasdadadsdadqw/MoS2_Mask_Rcnn/blob/master/json_to_coco.py.py)
* **Stage 1 Prediction Code (`mos2_predict.py`):** [Link](https://github.com/sadadadadasdadadsdadqw/MoS2_Mask_Rcnn/blob/master/mos2_predict.ipynb)
* **Stage 1 Evaluation Code (`mos2_pixel_evaluate.py`):** [Link](https://github.com/sadadadadasdadadsdadqw/MoS2_Mask_Rcnn/blob/master/mos2_pixel_evaluate.py)
* **Stage 2 Ablation Study Code:** [Link](https://github.com/sadadadadasdadadsdadqw/MoS2_Mask_Rcnn/tree/master/Ablation%20study)
* **Stage 3 DF-Image Processing (`BF-DF image fusion.py`):** [Link](https://github.com/sadadadadasdadadsdadqw/MoS2_Mask_Rcnn/blob/master/BF-DF%20image%20fusion.py)

---

## 1. Environment Preparation

The following environment configuration covers Ubuntu, PyCharm, and Anaconda setups, providing a ready-to-use debugging environment.

### Installation Steps
Configure your Python environment using the following commands:
* `conda create -n MoS2_Mask_Rcnn python=3.6`
* `conda activate MoS2_Mask_Rcnn`
* `pip install tensorflow-gpu==2.5.0`
* `pip install matplotlib`
* `pip install opencv-python==4.5.4.60`
* `pip install imgaug`
* `pip install pycocotools`
* `pip install pandas==1.1.5`
* `pip install scikit-learn==0.24.2`
* `pip install PyQt5==5.15.4`
* `pip install labelme==4.5.6`

---

## 2. Stage 1: Mask R-CNN Model

### 2.1 Basic Training Conditions
**Table S1. Basic conditions used in deep learning model training (including stages 1 and 2)**

| Parameter | Value |
| :--- | :--- |
| Backbone network | ResNet-101 |
| Optimizer | SGD with Momentum (β = 0.9) |
| Initial learning rate | 1 × 10⁻³ |
| Weight decay | 2 × 10⁻⁴ |
| Batch size | 1 |
| Training epoch | 400 |
| Steps per epoch | 500 |
| RPN Anchor scales | (32, 64, 128, 256, 512) |
| Learning rate scheduling | Constant |
| Image data augmentation | Random flip (H/V 70%), Rotation (±100°), Translation (±20%), Scaling (0.7-1.3), Shearing (±16), HSV adjustment |

### 2.2 Training Execution
Before starting, ensure the Python environment, pre-trained weights (**`[INSERT URL HERE: Pre-trained Weights]`**), and the MoS2 dataset are correctly configured.

**Steps:**
1.  Place `mos2_train.py` in the root directory. Create a `MoS2/` folder to store the input microscope images.
2.  Inside `MoS2/`, create `train2025/` and `val2025/` folders for the training and validation Bright Field (BF) images, respectively. Place the COCO-formatted JSON annotation files into the `MoS2/annotations/` folder.
3.  The core parameters are initialized in the `CocoConfig` class within `mos2_train.py`. `NUM_CLASSES` is set to 4 (background, single layer, contaminants, multi layer). To optimize for MoS2 microscope image segmentation, the `mrcnn_mask_loss` weight is increased to 2.0 to ensure edge segmentation accuracy.
4.  Execute the training command:
    `python mos2_train.py train --dataset=./MoS2 --model=resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5`
5.  During training, the `iaa.SomeOf` module dynamically applies data augmentation (random horizontal/vertical flips, affine transformations like rotation and scaling, and HSV color space adjustments). The model saves weights every 2 epochs.
6.  After 400 epochs, the final `.h5` weights are saved in an automatically generated timestamped subfolder under `logs/`.
7.  Performance metrics (Train Loss and Val Loss trends) are output as PNG, Excel, and CSV files in the automatically created `training_visualization/` folder.

### 2.3 Prediction Execution
The environment configuration for prediction is identical to the training environment.

**Steps:**
1.  Create a `Bimage/` folder in the root directory and place the target MoS2 BF images (.jpg, .jpeg, or .png) inside.
2.  Load the best weights saved from Stage 1 (epoch 400: **`[INSERT URL HERE: Epoch 400 Weights]`**) by renaming the file to `mask_rcnn_MoS2_400.h5` and placing it in the project root directory.
3.  In the prediction script, the confidence threshold (`DETECTION_MIN_CONFIDENCE`) is set to 0.7. Predictions with a probability lower than 0.7 are filtered out.
4.  Execute the prediction command:
    `python mos2_predict.py`
5.  The script will generate three folders:
    * `result/`: Contains the visualized color images with bounding boxes and labels (single layer, contaminants, multi layer).
    * `masks/`: Contains pure black-and-white binary foreground masks.
    * `masks2/`: Contains multi-layer segmentation masks.

### 2.4 Performance Evaluation
To evaluate the model, run the default test set using the following command:
`python "mos2_pixel_evaluate.py"`

---

## 3. Stage 2: Ablation Study

An ablation study was conducted to verify the optimal performance of different mask head architectures.

**Steps:**
1.  Replace `model.py` and `config.py` in the root `mrcnn` folder with the corresponding files from the Ablation study subfolders.
2.  Train the model following the same steps as Stage 1.
3.  After training, retrieve the new weights (**`[INSERT URL HERE: Ablation Study Weights]`**) and update the `COCO_MODEL_PATH` in `mos2_pixel_evaluate.py`.
4.  Run `python "mos2_pixel_evaluate.py"` to generate evaluation metrics. Results will be saved in the `evaluation_results/` folder.

**Table S2. Ablation study of different mask head architectures**

| Mask Head Architecture | AP | AR | F-1 | IOU |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (4 Conv + 1 Transposed Conv) | 0.983 | 0.983 | 0.958 | 0.920 |
| Variant A (4 Conv + 2 Bilinear Up) | 0.963 | 0.966 | 0.964 | 0.932 |
| Variant B (6 Conv + 1 Bilinear Up) | 0.985 | 0.933 | 0.958 | 0.920 |
| This work (6 Conv + 2 Bilinear Up) | 0.972 | 0.964 | 0.967 | 0.937 |

---

## 4. Stage 3: DF-Image Processing Details

This stage executes the fusion processing of bright-field (BF) and dark-field (DF) images using the `BF-DF image fusion.py` script.

### 4.1 Image Fusion Core Parameters
**Table S3. Fusion Parameters**

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Blue_threshold | 80 | Blue channel binarization threshold |
| CLIP_limit | 3.0 | CLAHE contrast limit |
| Tile_grid_size | (8, 8) | CLAHE grid size |

### 4.2 Execution Steps
1.  Ensure the following four folders exist in the root directory alongside `BF-DF image fusion.py`:
    * `Dimage/`: Contains original Dark Field (DF) images (.jpeg).
    * `Bimage/`: Contains original Bright Field (BF) images (.jpeg, corresponding 1:1 with DF images).
    * `masks/`: Contains binary foreground masks generated from Stage 1 (.png).
    * `masks2/`: Contains multi-layer segmentation masks generated from Stage 1 (.png).
2.  Execute the fusion script:
    `python "BF-DF image fusion.py"`
3.  The pipeline will automatically create a `process_result/` folder containing five subfolders: channel separation and enhancement results, morphological filling results, difference extraction maps, blended color overlays, and final refined masks.