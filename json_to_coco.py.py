import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
from PIL import ImageDraw
import os


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./tran.json'):
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            print(f"Processing file: {json_file} ({num + 1}/{len(self.labelme_json)})")
            try:
                with open(json_file, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    image_info = self.image(data, num)
                    if image_info is None:
                        continue
                    self.images.append(image_info)

                    for shapes in data['shapes']:
                        label = shapes['label']
                        if label not in self.label:
                            self.categories.append(self.categorie(label))
                            self.label.append(label)

                        points = shapes['points']
                        shape_type = shapes.get('shape_type', 'polygon')

                        # Process different types of annotations
                        if shape_type == 'rectangle' and len(points) == 2:
                            # Convert rectangle annotation to four points
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)

                            # Points in clockwise order
                            points = [
                                [x_min, y_min],
                                [x_max, y_min],
                                [x_max, y_max],
                                [x_min, y_max]
                            ]

                        annotation = self.annotation(points, label, num)
                        if annotation is not None:
                            self.annotations.append(annotation)
                            self.annID += 1
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
                continue

    def image(self, data, num):
        image = {}
        try:
            # Try loading image from imageData
            if 'imageData' in data and data['imageData'] is not None:
                img = utils.img_b64_to_arr(data['imageData'])
                height, width = img.shape[:2]
            else:
                # Load from image path
                image_path = data['imagePath']
                # If the path is relative, try to find it in the directory where the JSON file is located
                if not os.path.isabs(image_path):
                    json_dir = os.path.dirname(self.labelme_json[num])
                    image_path = os.path.join(json_dir, image_path)

                if not os.path.exists(image_path):
                    print(f"Warning: Image file does not exist: {image_path}")
                    return None

                img = cv2.imread(image_path)
                if img is None:
                    # Try opening with PIL
                    try:
                        img = np.array(PIL.Image.open(image_path))
                    except:
                        print(f"Error: Unable to load image: {image_path}")
                        return None

                height, width = img.shape[:2]

            image['height'] = height
            image['width'] = width
            image['id'] = num + 1
            image['file_name'] = os.path.basename(data['imagePath'])

            self.height = height
            self.width = width

            return image
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = 'Cancer'
        categorie['id'] = len(self.label) + 1
        categorie['name'] = label
        return categorie

    def annotation(self, points, label, num):
        try:
            annotation = {}
            annotation['segmentation'] = [list(np.asarray(points).flatten())]
            annotation['iscrowd'] = 0
            annotation['image_id'] = num + 1

            bbox = self.getbbox(points)
            if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                print(f"Warning: Invalid bounding box: {bbox}")
                return None

            annotation['bbox'] = bbox
            annotation['area'] = bbox[2] * bbox[3]
            annotation['category_id'] = self.getcatid(label)
            annotation['id'] = self.annID
            return annotation
        except Exception as e:
            print(f"Error creating annotation: {e}")
            return None

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1

    def getbbox(self, points):
        """Calculate bounding box directly through point coordinates, more efficient"""
        try:
            points = np.array(points)
            x_min = np.min(points[:, 0])
            y_min = np.min(points[:, 1])
            x_max = np.max(points[:, 0])
            y_max = np.max(points[:, 1])

            width = x_max - x_min
            height = y_max - y_min

            # Ensure bounding box is valid
            if width <= 0 or height <= 0:
                print(f"Warning: Invalid bounding box size: {width}x{height}")
                return None

            return [x_min, y_min, width, height]
        except Exception as e:
            print(f"Error calculating bounding box: {e}")
            return None

    def polygons_to_mask(self, img_shape, polygons):
        """Convert polygon to mask (fallback method)"""
        try:
            mask = np.zeros(img_shape, dtype=np.uint8)
            polygons = np.array(polygons, dtype=np.int32)
            cv2.fillPoly(mask, [polygons], 1)
            return mask
        except:
            # If cv2 fails, use PIL method
            try:
                mask = np.zeros(img_shape, dtype=np.uint8)
                mask = PIL.Image.fromarray(mask)
                xy = list(map(tuple, polygons))
                ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
                mask = np.array(mask, dtype=np.uint8)
                return mask
            except Exception as e:
                print(f"Error creating mask: {e}")
                return None

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        print("Starting conversion of Labelme annotations to COCO format...")
        self.data_transfer()

        if not self.images:
            print("Error: No images were successfully processed")
            return

        if not self.annotations:
            print("Warning: No annotations found")

        self.data_coco = self.data2coco()

        # Save JSON file
        try:
            with open(self.save_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.data_coco, f, indent=4, cls=MyEncoder, ensure_ascii=False)
            print(f"Conversion complete! COCO format annotations saved to: {self.save_json_path}")
            print(f"Statistics:")
            print(f"  Number of images: {len(self.images)}")
            print(f"  Number of categories: {len(self.categories)}")
            print(f"  Number of annotations: {len(self.annotations)}")
            print("Category list:")
            for cat in self.categories:
                print(f"  - {cat['name']} (id: {cat['id']})")
        except Exception as e:
            print(f"Error saving JSON file: {e}")


# Usage example
if __name__ == "__main__":
    # Get all JSON files
    labelme_json = glob.glob('C:\\Users\\31227\\Desktop\\train\\*.json')

    if not labelme_json:
        print("Error: No JSON files found")
        exit(1)

    print(f"Found {len(labelme_json)} JSON files")

    # Convert
    labelme2coco(labelme_json, 'C:\\Users\\31227\\Desktop\\instances_train2019.json')