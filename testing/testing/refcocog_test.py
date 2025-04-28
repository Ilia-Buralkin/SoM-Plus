import json
import os
import sys
import base64
import io
import pandas as pd
from PIL import Image
import re
from tqdm import tqdm
import warnings
import argparse
from dotenv import load_dotenv
import textwrap
# Import VertexAI libraries instead of google.generativeai
import traceback
from google.auth import credentials as auth_credentials
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part
from vertexai.preview import generative_models
import vertexai
from som_plus.gemini_vlm_query import perform_visual_grounding

warnings.simplefilter(action='ignore', category=FutureWarning)

env_path = "/home/iliab/rice/final_project/.env"
load_dotenv(dotenv_path=env_path)


from shapely.geometry import Polygon, MultiPolygon

import io
import time
import threading
import traceback
import re
from PIL import Image
from vertexai.generative_models import Part
from vertexai.preview import generative_models
import vertexai

from som_plus.gemini_vlm_query import perform_visual_grounding


def rescale_bounding_box(bbox, orig_size, new_size):
    """
    Rescale a bounding box from original image size to a new image size.

    Args:
        bbox (list): [x, y, width, height]
        orig_size (tuple): (orig_width, orig_height)
        new_size (tuple): (new_width, new_height)

    Returns:
        list: Rescaled bounding box [x', y', w', h']
    """
    orig_w, orig_h = orig_size
    new_w, new_h = new_size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    x, y, w, h = bbox
    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]

def calculate_iou(gt_box, pred_box):
    gt_x, gt_y, gt_w, gt_h = gt_box
    pred_x, pred_y, pred_w, pred_h = pred_box

    # Compute corners
    gt_x1, gt_y1 = gt_x, gt_y
    gt_x2, gt_y2 = gt_x + gt_w, gt_y + gt_h

    pred_x1, pred_y1 = pred_x, pred_y
    pred_x2, pred_y2 = pred_x + pred_w, pred_y + pred_h

    # Intersection box
    inter_x1 = max(gt_x1, pred_x1)
    inter_y1 = max(gt_y1, pred_y1)
    inter_x2 = min(gt_x2, pred_x2)
    inter_y2 = min(gt_y2, pred_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    gt_area = gt_w * gt_h
    pred_area = pred_w * pred_h
    union_area = gt_area + pred_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def is_center_inside(gt_box, pred_box):
    gt_x, gt_y, gt_w, gt_h = gt_box
    pred_x, pred_y, pred_w, pred_h = pred_box

    # Predicted center
    pred_center_x = pred_x + pred_w / 2
    pred_center_y = pred_y + pred_h / 2

    # GT bounds
    gt_xmin = gt_x
    gt_xmax = gt_x + gt_w
    gt_ymin = gt_y
    gt_ymax = gt_y + gt_h

    return (gt_xmin <= pred_center_x <= gt_xmax) and (gt_ymin <= pred_center_y <= gt_ymax)

def flatten_to_coords(flat_list):
    return [(flat_list[i], flat_list[i+1]) for i in range(0, len(flat_list), 2)]

def rescale_polygon(polygon, orig_size, new_size):
    """
    Scale a polygon from orig_size to new_size.
    
    Args:
        polygon (list of (x, y)): Polygon points
        orig_size (tuple): (orig_width, orig_height)
        new_size (tuple): (new_width, new_height)
        
    Returns:
        list of (x, y): Scaled polygon
    """
    orig_w, orig_h = orig_size
    new_w, new_h = new_size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    return [(x * scale_x, y * scale_y) for (x, y) in polygon]

def process_segmentation(segmentation_data):
    """
    Process segmentation data that may contain multiple polygons
    
    Args:
        segmentation_data: List of lists, where each inner list is a polygon
        
    Returns:
        Shapely geometry (MultiPolygon or Polygon)
    """
    from shapely.geometry import Polygon, MultiPolygon
    
    # Check if we have multiple polygons
    if len(segmentation_data) > 1:
        polygons = []
        for polygon_coords in segmentation_data:
            # Convert flat list to coordinates
            coords = flatten_to_coords(polygon_coords)
            # Create a Shapely polygon
            poly = Polygon(coords)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
        
        # If we have multiple valid polygons, return a MultiPolygon
        if len(polygons) > 1:
            return MultiPolygon(polygons)
        # If we only have one valid polygon, return it
        elif len(polygons) == 1:
            return polygons[0]
        # If no valid polygons, return an empty polygon
        else:
            return Polygon()
    else:
        # Single polygon case
        coords = flatten_to_coords(segmentation_data[0])
        return Polygon(coords)




def calculate_iou_with_multipolygons(
        gt_segmentation, output_segmentation,
        orig_width, orig_height, 
        new_width, new_height):
    """
    Calculate IoU between potentially multi-polygon segmentations
    
    Args:
        gt_segmentation: Ground truth segmentation data
        output_segmentation: Predicted segmentation data
        
    Returns:
        float: IoU value between 0 and 1
    """
    
    try:
        # Process both segmentations
        gt_geom = Polygon(rescale_polygon(flatten_to_coords(gt_segmentation[0]), (orig_width, orig_height), (new_width, new_height)))
        output_geom = process_segmentation(output_segmentation)


        # Fix any invalid geometries
        if not gt_geom.is_valid:
            gt_geom = gt_geom.buffer(0)

        if not output_geom.is_valid:
            output_geom = output_geom.buffer(0)
        
        # Calculate IoU
        intersection_area = gt_geom.intersection(output_geom).area
        union_area = gt_geom.union(output_geom).area
        
        if union_area < 1e-10:  # Avoid division by zero
            return 0.0
            
        iou = intersection_area / union_area
        return iou
        
    except Exception as e:
        print(f"Error in IoU calculation: {e}")
        return 0.0



def test_som_combined(data, labeled_data_path, model, method, iou_threshold=0.5, vlm=True):
    """
    Test SoM and SoM+ methods on RefCOCOg dataset.
    Keeping this function as close to the original as possible.
    
    Args:
        data (list): List of image data dictionaries
        labeled_data_path (str): Path to labeled images
        grounding_vlm (VertexVisualGrounding): VLM instance
        aug (bool): Whether to use augmented descriptions
        single (bool): Whether to use unified descriptions
        iou_threshold (float): IoU threshold for evaluation
        vlm (bool): Whether to use VLM or LLM only
        
    Returns:
        pd.DataFrame: Results dataframe
    """
    # Load original annotations for reference
    og_js_path = "/home/iliab/rice/final_project/refcocog_dataset/refcocog_refseg/orig/refcocog/instances.json"
    images_w_annotations = json.load(open(og_js_path, "r"))

    df = pd.DataFrame(columns=[
        'image_id', 'sentences', 'choice',
        'gt_box', 'pred_box', 'box_iou', 'center_match',
        'gt_mask', 'output_mask', 'mask_iou',
        'prompt'
    ])

    total_center_matches = 0
    total_iou_matches = 0

    for meta_data in tqdm(data):
        question_meta = meta_data["question"]
        image_meta = meta_data["image_info"]
        annotations_meta = meta_data["annotations_info"]

        file_name = image_meta['file_name']
        for i in images_w_annotations['images']:
            if i['file_name'] == file_name:
                orig_height, orig_width = i['height'], i['width']

        image_id = str(image_meta['id'])
        sentences = [s['raw'] for s in question_meta['sentences']]
        ann_id = question_meta['ann_id']

        labeled_image_path = labeled_data_path + image_id + ".jpg"
        labeled_image = Image.open(labeled_image_path)
        new_w, new_h = labeled_image.size

        labels_path = labeled_data_path + image_id + ".json"
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        boxes = [list(i['bbox']) for i in labels]
        masks = [i['segmentation'] for i in labels] 
        descriptions = []

        if method == "unified":
            with open(labeled_data_path + image_id + "_pro_single.txt", "r") as f:
                descriptions = f.read()
        elif method == "parallel":
            descriptions = [i['description'] for i in labels]
        
        output_id, prompt = perform_visual_grounding(question = sentences, image = labeled_image,
                                                     descriptions = descriptions, model = model,
                                                     mode = method, vlm = vlm)
        
        # Load ground truth bounding box and mask
        gt_box = None
        gt_mask = None
        for ann in annotations_meta:
            if ann['id'] == ann_id:
                gt_box = ann['bbox']
                gt_mask = ann['segmentation'][0]
                break

        pred_box = None
        center_match = False
        box_iou = None

        output_mask = None
        mask_iou = None

        if output_id is not None and 1 <= output_id <= len(boxes):
            # Bounding Box evaluation
            pred_box = boxes[output_id - 1]
            gt_box_rescaled = rescale_bounding_box(gt_box, (orig_width, orig_height), (new_w, new_h))
            box_iou = calculate_iou(gt_box_rescaled, pred_box)
            center_match = is_center_inside(gt_box_rescaled, pred_box)
            if center_match:
                total_center_matches += 1

            # Mask evaluation
            output_mask = masks[output_id - 1]
            mask_iou = calculate_iou_with_multipolygons(
                [gt_mask], output_mask,
                orig_width, orig_height,
                new_w, new_h
            )
            if mask_iou is not None and mask_iou > iou_threshold:
                total_iou_matches += 1

        new_row = pd.DataFrame([{
            'image_id': image_id,
            'sentences': sentences,
            'choice': output_id,
            'gt_box': gt_box,
            'pred_box': pred_box,
            'box_iou': box_iou,
            'center_match': center_match,
            'gt_mask': gt_mask,
            'output_mask': output_mask,
            'mask_iou': mask_iou,
            'prompt': prompt
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    print("Box Accuracy (center match):", total_center_matches / len(data))
    print("Box Accuracy (IoU > {:.2f}):".format(iou_threshold), df.loc[df['box_iou'] > iou_threshold].shape[0] / len(data))
    print("Mask Accuracy (IoU > {:.2f}):".format(iou_threshold), total_iou_matches / len(data))

    return df



def main():
    parser = argparse.ArgumentParser(description="Test RefCOCOg with SoM and SoM+ using VertexAI")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", 
                        choices=["gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-pro-preview-03-25"],
                        help="Model to use")
    parser.add_argument("--segmentation", type=str, default="maskformer", 
                        choices=["maskformer", "sam2"],
                        help="Segmentation model")
    parser.add_argument("--method", type=str, default="baseline", 
                        choices=["baseline", "parallel", "unified"],
                        help="SoM method to use")
    parser.add_argument("--no_vlm", default=False, action="store_true",
                        help="Use VLM (True) or LLM (False)")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/iliab/rice/final_project/clean_notebooks/clean_resutls/final/refcocog/", 
                        help="Output directory")
    parser.add_argument("--qps", type=float, default=1.0, help="Adjust the rate for Gemini api (1.5 Pro)")
    args = parser.parse_args()
    
    vlm = (args.no_vlm == False)

    # Set up paths based on your original structure
    if args.segmentation == "maskformer":
        labeled_data_path = "/home/iliab/rice/final_project/refcocog_dataset/selected_data/labeled_images_v4/maskformer/"
    else:  # sam2
        labeled_data_path = "/home/iliab/rice/final_project/refcocog_dataset/selected_data/labeled_images_v4/sam2/"
    
    print("labeled_data_path: ", labeled_data_path)
    test_data_path = "/home/iliab/rice/final_project/refcocog_dataset/selected_data/val_images_250/test_data.json"
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    
    # Run test
    results = test_som_combined(
        data=test_data[:],
        labeled_data_path=labeled_data_path,
        model = args.model, 
        method = args.method,
        iou_threshold=0.5,
        vlm=vlm
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_str = args.model.replace('-', '_')
    seg_str = args.segmentation
    method_str = args.method

    vlm_str = "vlm" if vlm else "llm"
    
    filename = f"results_{model_str}_{seg_str}_{method_str}_{vlm_str}.csv"
    results.to_csv(os.path.join(args.output_dir, filename), index=False)
    
    print(f"Results are saved to {args.output_dir + filename}")


if __name__ == "__main__":
    main()
