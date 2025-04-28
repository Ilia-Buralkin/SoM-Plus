import cv2
import numpy as np
from PIL import Image
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
import os
import json
from tqdm import tqdm
from pycocotools import mask as mask_utils
from som_plus.augmentation import augment_annotations_unified, augment_annotations_parallel
from som_plus.segmentation_pipeline import SegmentationPipelineFactory
import os
import os
import argparse
import pandas as pd
import sys

sys.path.append("/home/iliab/rice/final_project/packages/sam2/") # define the path to the sam2 package
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv("/home/iliab/rice/final_project/.env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def mask_to_polygons(mask):
    """
    Converts binary mask to polygon format (COCO style).
    
    Args:
        mask: np.ndarray of shape (H, W), binary mask.
    
    Returns:
        polygons: list of list of floats
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for contour in contours:
        if len(contour) >= 3:  # valid polygon
            poly = contour.reshape(-1).astype(float).tolist()
            polygons.append(poly)

    return polygons

def polygon_to_mask(polygons, height, width):
    """
    Converts COCO polygon format to binary mask.
    
    Args:
        polygons: list of list of floats (COCO polygon format).
        height: height of the target mask.
        width: width of the target mask.
    
    Returns:
        mask: np.ndarray of shape (height, width), dtype=uint8
    """
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask

def serialize_annotation_to_polygons(anns):
    """Convert binary masks to polygon format and clean other fields."""
    serialized = []
    for ann in anns:
        new_ann = ann.copy()

        if isinstance(new_ann["segmentation"], np.ndarray):
            polygons = mask_to_polygons(new_ann["segmentation"])
            new_ann["segmentation"] = polygons

        if hasattr(new_ann["bbox"], "tolist"):
            new_ann["bbox"] = new_ann["bbox"].tolist()

        # Keep only if we successfully extracted polygons
        if new_ann["segmentation"]:
            serialized.append(new_ann)
    return serialized

def define_pipeline(pipeline_name):

    if pipeline_name == "maskformer":
        maskformer_model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-coco")
        maskformer_processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-coco")

        maskformer_config = {
            "threshold": 0.95,
            "mask_area_threshold": 0.9,
            "mask_fill_alpha": 0.05,
            "mask_border_thickness": 2,
            "remove_small": True,
            "area_thresh": 100,
            "remove_mode": "islands"
        }

        # Create MaskFormer pipeline
        maskformer_pipeline = SegmentationPipelineFactory.create_pipeline(
            model_type="maskformer",
            model=maskformer_model,
            processor=maskformer_processor,
            config=maskformer_config
        )
        return maskformer_pipeline

    if pipeline_name == "sam2":

        checkpoint = "/home/iliab/rice/final_project/packages/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        sam2 = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))


        sam2_config = {
            "grid_stride": 50,
            "threshold": 0.8,
            "mask_area_threshold": 0.5,
            "mask_fill_alpha": 0.05,
            "mask_border_thickness": 2,
            "remove_small": True,
            "area_thresh": 100,
            "remove_mode": "islands",
            "iou_thresh": 0.7,
            "contain_thresh": 0.75,
            "refine_boundaries": True,
            "overlap_threshold": 0.2
        }

        sam2_pipeline = SegmentationPipelineFactory.create_pipeline(
            model_type="sam2",
            model=sam2,
            config=sam2_config
        )
        return sam2_pipeline

def gqa_test_prep(
    df, 
    image_dir: str, 
    pipeline_name: str,
    output_dir: str = "/home/iliab/rice/final_project/gqa_dataset/labeled_images",
    max_workers: int = 8,
    n: int = 250,
    grayscale_background: bool = True,
    blur_strength: int = 0,
    show_label: bool = False
) -> pd.DataFrame:
    """
    Create annotations for images in a dataframe using segmentation pipeline
    and Gemini models for description generation. Generates both single-pass and
    per-object annotations simultaneously.
    
    Args:
        df: DataFrame containing image information
        image_dir: Directory containing input images
        pipeline: Segmentation pipeline object
        output_dir: Directory to save processed results
        unified_aug_model: Model for unified (single-pass) augmentation (Gemini Pro)
        parallel_aug_model: Model for parallel augmentation (Gemini Flash)
        max_tokens_unified: Maximum tokens for unified augmentation model
        max_tokens_parallel: Maximum tokens for parallel augmentation model
        n: Number of images to process (None for all)
        max_workers: Number of parallel workers for processing
        grayscale_background: Whether to grayscale the areas outside the mask
        blur_strength: Strength of blur to apply to background (0 for no blur)
        show_label: Whether to show labels on the visualized annotations
        
    Returns:
        DataFrame with added descriptions (both single and multiple)
    """

    pipeline = define_pipeline(pipeline_name = pipeline_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit the dataframe if n is specified
    if n is not None:
        df = df.copy().iloc[:n]
    else:
        df = df.copy()
    
    # Create columns for both types of descriptions
    df['single_description'] = None
    df['multiple_descriptions'] = None
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row['image_name']
        image_path = os.path.join(image_dir, image_name)
        
        try: 
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue
            
        # Run pipeline
        annotated_image, annotations = pipeline.segment_and_annotate(image)
        
        single_description = augment_annotations_unified(
                annotated_image = annotated_image,
                annotations=annotations, 
                image_id = image_name
                )

        augmented_anns = augment_annotations_parallel(
            pipeline=pipeline,
            image=image, 
            annotations=annotations, 
            max_workers=max_workers,
            grayscale_background=grayscale_background, 
            blur_strength=blur_strength,
            show_label = show_label
        )

        df.at[i, "single_description_pro"] = single_description
        
        multiple_descriptions = [ann['description'] for ann in augmented_anns]
        df.at[i, "multiple_descriptions"] = multiple_descriptions
        
        # Save annotated image
        annotated_image.save(os.path.join(output_dir, image_name))
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/iliab/rice/final_project/gqa_dataset/selected_queries.csv")
    parser.add_argument("--pipeline_name", type=str, default="maskformer", choices=['maskformer', 'sam2'])
    parser.add_argument("--image_dir", type=str, default="/home/iliab/rice/final_project/gqa_dataset/images")
    parser.add_argument("--output_dir", type=str, default="/home/iliab/rice/final_project/gqa_dataset/labeled_images_v3/check")
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--number_of_queries", type=int, default = 250)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path) 
    
    processed_data = gqa_test_prep(
            df = data,
            image_dir = args.image_dir,
            pipeline_name = args.pipeline_name,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            n=args.number_of_queries
            )
    
    processed_data.to_csv(args.ouput_dir + "/selected_queries_processed.csv")

    return "Complete..."

if __name__ == "__main__":
    main()
