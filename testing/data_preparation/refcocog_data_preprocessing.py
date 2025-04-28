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



def refcocog_test_prep(
    data_path: str,
    pipeline_name: str,
    image_dir: str,
    output_dir: str,
    max_workers: int,
    grayscale_rest: bool = True, 
    blur_strength: int = 0, 
    show_label: bool = True,
) -> None:
    """
    Prepare and process RefCOCOg test data by:
    1. Running images through segmentation pipeline
    2. Generating descriptions using Gemini models (both unified and parallel approaches)
    3. Saving annotated images and descriptions
    
    Args:
        data: List of data items from RefCOCOg dataset
        pipeline: Segmentation pipeline object
        image_dir: Directory containing input images
        output_dir: Directory to save processed results
        unified_aug_model: Model for unified (single-pass) augmentation (Gemini Pro)
        unified_aug_model2: Second model for unified augmentation (Gemini Flash)
        parallel_aug_model: Model for parallel augmentation (Gemini Flash)
        max_tokens_unified: Maximum tokens for unified augmentation model
        max_tokens_parallel: Maximum tokens for parallel augmentation model
        max_workers: Number of parallel workers for processing
    """
    with open(data_path) as f:
        data = json.load(f)
    
    pipeline = define_pipeline(pipeline_name = pipeline_name) 

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    
    for item in tqdm(data):
        image_info = item["image_info"]
        image_id = image_info["id"]
        question_info = item["question"]
        question = question_info["sentences"]
        
        # Load the image
        try:
            image = Image.open(os.path.join(image_dir, f"{image_id}.jpg")).convert("RGB")
        except Exception as e:
            print(f"Failed to open image {image_id}: {e}")
            continue
            
        # Run pipeline
        annotated_img, anns = pipeline.segment_and_annotate(image)
        
        
        descriptions = augment_annotations_unified(
                annotated_image = annotated_img,
                annotations=anns, 
                image_id = image_id
                )

        augmented_ann = augment_annotations_parallel(
            pipeline=pipeline,
            image=image, 
            annotations=anns, 
            max_workers=max_workers,
            grayscale_background=grayscale_rest, 
            blur_strength=blur_strength,
            show_label = show_label
        )
        
        # Save annotated image
        image_save_path = os.path.join(output_dir, f"{image_id}.jpg")
        annotated_img.save(image_save_path)
        
        # Save single-pass descriptions
        single_ann_save_path = os.path.join(output_dir, f"{image_id}_pro_single.txt")
        with open(single_ann_save_path, "w") as f:
            f.write(descriptions)

        # Save per-annotation descriptions
        ann_save_path = os.path.join(output_dir, f"{image_id}.json")
        serialized_anns = serialize_annotation_to_polygons(augmented_ann)
        with open(ann_save_path, "w") as f:
            json.dump(serialized_anns, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/iliab/rice/final_project/refcocog_dataset/selected_data/val_images_250/test_data.json")
    parser.add_argument("--pipeline", type=str, default="maskformer", choices=['maskformer', 'sam2'])
    parser.add_argument("--image_dir", type=str, default="/home/iliab/rice/final_project/refcocog_dataset/selected_data/val_images_250")
    parser.add_argument("--output_dir", type=str, default="/home/iliab/rice/final_project/refcocog_dataset/selected_data/labeled_images_v4/maskformer/check")
    parser.add_argument("--max_workers", type=int, default=10)
    args = parser.parse_args()


    refcocog_test_prep(
            data_path = args.data,
            pipeline_name=args.pipeline,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            max_workers=args.max_workers
            )

    return "Complete..."

if __name__ == "__main__":
    main()
