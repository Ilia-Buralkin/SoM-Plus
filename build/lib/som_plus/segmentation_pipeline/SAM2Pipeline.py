# sam2_pipeline.py

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple

from .AbstractSegmentationPipeline import AbstractSegmentationPipeline

class SAM2Pipeline(AbstractSegmentationPipeline):
    """
    A concrete implementation of the segmentation pipeline for SAM2 models.
    Runs a SAM2 model for panoptic segmentation on images.
    """

    def __init__(
        self,
        predictor,  # SAM2ImagePredictor 
        grid_stride: int = 64,
        threshold: float = 0.8,
        mask_area_threshold: float = 0.5,
        mask_fill_alpha: float = 0.3,
        mask_border_thickness: int = 2,
        remove_small: bool = False,
        area_thresh: int = 100,
        remove_mode: str = "islands",
        iou_thresh: float = 0.7,
        contain_thresh: float = 0.99,
        refine_boundaries: bool = True,
        overlap_threshold: float = 0.2,
    ):
        """
        Constructor for SAM2Pipeline.

        Args:
            predictor: A SAM2ImagePredictor instance.
            grid_stride (int): Stride for point grid sampling.
            threshold (float): Confidence threshold for predictions.
            mask_area_threshold (float): Threshold for overlap mask area.
            mask_fill_alpha (float): Opacity for mask fill on the annotated image.
            mask_border_thickness (int): Thickness in pixels for mask borders.
            remove_small (bool): Whether to remove small regions from masks.
            area_thresh (int): Minimum area threshold for small region removal.
            remove_mode (str): Either "islands" (remove small blobs) or "holes" (fill holes).
            iou_thresh (float): IOU threshold for filtering overlapping masks.
            contain_thresh (float): Containment threshold for filtering nested masks.
            refine_boundaries (bool): Whether to refine boundaries between overlapping masks.
            overlap_threshold (float): Threshold for significant overlap in boundary refinement.
        """
        super().__init__(
            threshold=threshold,
            mask_area_threshold=mask_area_threshold,
            mask_fill_alpha=mask_fill_alpha,
            mask_border_thickness=mask_border_thickness,
            remove_small=remove_small,
            area_thresh=area_thresh,
            remove_mode=remove_mode
        )
        self.predictor = predictor
        self.grid_stride = grid_stride
        self.iou_thresh = iou_thresh
        self.contain_thresh = contain_thresh
        self.refine_boundaries = refine_boundaries
        self.overlap_threshold = overlap_threshold

    def segment_and_annotate(self, image: Image.Image):
        """
        Main entry point for SAM2:
        1) Runs SAM2 segmentation on an input PIL image with grid point prompts,
        2) Filters and refines the masks,
        3) Optionally removes small regions,
        4) Draws each mask + numeric label onto a copy of the image.

        Returns:
            annotated_image (PIL.Image):
                The input image with colored masks and numeric labels drawn.
            annotations (List[dict]):
                A list of annotation dicts for each detected mask, sorted by area.
        """
        # Convert to BGR for OpenCV operations
        image_bgr = self._pil_to_bgr(image)

        # Run SAM2 inference
        masks = self._run_inference(image)
        
        # Filter by area
        masks = [m for m in masks if m.sum() > self.area_thresh]
        
        # Filter overlapping and apply boundary refinement
        masks = self._filter_nested_and_overlapping_masks(masks)
        
        # Remove small regions if needed
        if self.remove_small:
            processed_masks = []
            for mask in masks:
                cleaned_mask, _ = self._remove_small_regions(
                    mask,
                    area_thresh=self.area_thresh,
                    mode=self.remove_mode
                )
                processed_masks.append(cleaned_mask)
            masks = processed_masks

        # Build annotations from masks
        annotations = self._generate_annotations(masks)

        # Draw each mask and label on the BGR image
        self._draw_masks_and_labels(image_bgr, masks, annotations)

        # Convert annotated BGR numpy array back to PIL
        annotated_image = self._bgr_to_pil(image_bgr)
        return annotated_image, annotations

    def _run_inference(self, image: Image.Image) -> List[np.ndarray]:
        """
        Runs the SAM2 model in "all-points" mode and returns a list of masks.

        Args:
            image (PIL.Image): The input image in PIL format.

        Returns:
            List[np.ndarray]: A list of binary masks.
        """
        image_np = np.asarray(image)
        self.predictor.set_image(image_np)
        points = self._generate_point_grid(image_np.shape)
        
        all_masks = []
        for pt in points:
            masks, _, _ = self.predictor.predict(
                point_coords=pt[None, :],
                point_labels=np.array([1]),
                multimask_output=False,
            )
            all_masks.extend([m.astype(bool) for m in masks])
        return all_masks

    def _generate_point_grid(self, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """Returns a dense grid of foreground points across the image."""
        h, w = image_shape[:2]
        ys, xs = np.meshgrid(np.arange(0, h, self.grid_stride), np.arange(0, w, self.grid_stride))
        return np.stack([xs.ravel(), ys.ravel()], axis=-1)

    def _filter_nested_and_overlapping_masks(
        self,
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Remove masks that are nested or highly overlapping with others."""
        masks = [m.astype(bool) for m in masks]
        keep, used = [], set()

        for i, m1 in enumerate(masks):
            if i in used:
                continue
            for j, m2 in enumerate(masks):
                if j == i or j in used:
                    continue
                inter = np.logical_and(m1, m2).sum()
                if inter == 0:
                    continue
                union = np.logical_or(m1, m2).sum()
                iou = inter / union
                contain_m2_in_m1 = inter / m2.sum()
                if iou > self.iou_thresh or contain_m2_in_m1 > self.contain_thresh:
                    used.add(j)
            keep.append(m1.astype(np.uint8))
            used.add(i)
        
        # After initial filtering, apply boundary refinement to remaining masks if enabled
        if self.refine_boundaries:
            refined_masks = self._refine_mask_boundaries(keep)
            return refined_masks
        return keep

    def _refine_mask_boundaries(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Refines boundaries between overlapping masks by subtracting overlapping 
        regions from the larger mask.
        
        Args:
            masks (List[np.ndarray]): List of binary masks to process.
            
        Returns:
            List[np.ndarray]: Refined masks with overlap conflicts resolved.
        """
        if not masks or len(masks) < 2:
            return masks
            
        # Convert to boolean masks for logical operations
        bool_masks = [m.astype(bool) for m in masks]
        areas = [m.sum() for m in bool_masks]
        refined_masks = [m.copy() for m in bool_masks]
        
        # For each pair of masks, check for overlap
        for i in range(len(bool_masks)):
            for j in range(i + 1, len(bool_masks)):
                # Calculate overlap
                overlap = np.logical_and(refined_masks[i], refined_masks[j])
                overlap_size = overlap.sum()
                
                # Skip if overlap is below threshold
                if overlap_size == 0 or overlap_size < self.overlap_threshold * min(areas[i], areas[j]):
                    continue
                    
                # Determine which mask is larger (by area)
                if areas[i] > areas[j]:
                    larger_idx, smaller_idx = i, j
                else:
                    larger_idx, smaller_idx = j, i
                
                # Subtract the overlap from the larger mask
                refined_masks[larger_idx] = np.logical_and(
                    refined_masks[larger_idx], 
                    ~overlap
                )
                
                # Update area after refinement
                areas[larger_idx] = refined_masks[larger_idx].sum()
        
        # Convert back to uint8 masks
        return [m.astype(np.uint8) for m in refined_masks]
