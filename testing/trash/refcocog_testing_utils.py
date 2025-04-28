# Standard library imports
import textwrap  # Already included in your file
import json
from typing import List, Tuple, Dict, Any, Optional

# Third-party imports
import numpy as np  # If used for numerical operations
from shapely.geometry import Polygon, MultiPolygon  # Used in your segmentation functions


def generate_prompt_baseline(sentences):
    prompt = f"""
    You are given an image with multiple annotated objects, each labeled with a numeric ID.
    The user's query: {sentences}
    Pick exactly one annotation ID that corresponds best to this query.
    Write it as: Annotation_ID_x
    Remember not to mention anything else.
    """
    return prompt

import textwrap

def generate_prompt_parallel_descriptions(query_sentences, annotation_descriptions):
    """
    Args:
        query_sentences (list[str]): A list of referring expression sentences (e.g., ["zebra in the middle"]).
        annotation_descriptions (list[str]): A list of text descriptions for each annotation, 
                                             each presumably containing the structured fields 
                                             (Description, Relative location, etc.).
    
    Returns:
        A formatted prompt (string) that clearly presents all annotation descriptions 
        and asks the model to pick one annotation ID for the given query.
    """

    # Join the user’s referring expression sentences (if there are multiple)
    query_str = " ".join(query_sentences).strip()

    # Build a readable multiline block of annotation descriptions, each on its own paragraph.
    # Since each description might already contain newline formatting, 
    # we'll just separate them with a blank line for clarity.
    descriptions_block = "\n\n".join(annotation_descriptions)

    # Create the final prompt
    prompt = f"""
You are given an image with multiple annotated objects, each labeled with a numeric ID. 
Below is a list of those object IDs and their descriptions:

{descriptions_block}

The user's query: "{query_str}"

Pick exactly one annotation ID that corresponds best to this query while utilizing the descriptions, and write it in this format:
Annotation_ID_x

Do not include any additional commentary. 
Only produce the line "Annotation_ID_x" and nothing else.
"""

    # Use textwrap.dedent to remove extra indentation, then strip leading/trailing whitespace
    prompt = textwrap.dedent(prompt).strip()
    return prompt

def generate_prompt_unified_descriptions(sentences, annotation_descriptions):
    prompt = f"""
    You are given an image with multiple annotated objects, each labeled with a numeric ID. 
Below is a list of those object IDs and their descriptions:

{annotation_descriptions}

The user's query: "{sentences}"

Pick exactly one annotation ID that corresponds best to this query while utilizing the descriptions, and write it in this format:
Annotation_ID_x

Do not include any additional commentary. 
Only produce the line "Annotation_ID_x" and nothing else.
    """
    return prompt

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
    from shapely.geometry import Polygon
    
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

