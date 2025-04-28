# maskformer_pipeline.py

import torch
import numpy as np
from PIL import Image
from typing import Tuple, List
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor

from .AbstractSegmentationPipeline import AbstractSegmentationPipeline

class MaskFormerPipeline(AbstractSegmentationPipeline):
    """
    A concrete implementation of the segmentation pipeline for MaskFormer models.
    Runs a MaskFormer model for panoptic segmentation on images.
    """

    def __init__(
        self,
        model: MaskFormerForInstanceSegmentation,
        processor: MaskFormerImageProcessor,
        threshold: float = 0.8,
        mask_area_threshold: float = 0.5,
        mask_fill_alpha: float = 0.3,
        mask_border_thickness: int = 2,
        remove_small: bool = False,
        area_thresh: int = 100,
        remove_mode: str = "islands",
    ):
        """
        Constructor for MaskFormerPipeline.

        Args:
            model (MaskFormerForInstanceSegmentation):
                A pre-trained MaskFormer model for instance or panoptic segmentation.
            processor (MaskFormerImageProcessor):
                The corresponding MaskFormerImageProcessor for preprocessing/postprocessing.
            threshold (float):
                Threshold for binarizing masks after inference.
            mask_area_threshold (float):
                Threshold for overlap mask area (postprocessing).
            mask_fill_alpha (float):
                Opacity for mask fill on the annotated image. Range [0.0 - 1.0].
            mask_border_thickness (int):
                Thickness in pixels for the drawn mask borders.
            remove_small (bool):
                Whether to remove small disconnected regions or holes in the predicted masks.
            area_thresh (int):
                Minimum area to retain or fill when removing small regions.
            remove_mode (str):
                Either "islands" (remove small blobs) or "holes" (fill small holes).
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
        self.model = model
        self.processor = processor

    def segment_and_annotate(self, image: Image.Image):
        """
        Main entry point for MaskFormer:
        1) Runs MaskFormer panoptic segmentation on an input PIL image,
        2) Optionally removes small regions,
        3) Draws each mask + numeric label onto a copy of the image.

        Returns:
            annotated_image (PIL.Image):
                The input image with colored masks and numeric labels drawn.
            annotations (List[dict]):
                A list of annotation dicts for each detected mask, sorted by area.
        """
        # Convert to BGR for OpenCV operations
        image_bgr = self._pil_to_bgr(image)

        # Run MaskFormer inference
        seg_map, segments_info = self._run_inference(image)

        # Build the masks (Boolean tensors), optionally remove small components
        masks = self._build_masks(seg_map, segments_info)

        # Create bounding boxes, etc., from masks
        annotations = self._generate_annotations(masks)

        # Draw each mask and label on the BGR image
        self._draw_masks_and_labels(image_bgr, masks, annotations)

        # Convert annotated BGR numpy array back to PIL
        annotated_image = self._bgr_to_pil(image_bgr)
        return annotated_image, annotations

    def _run_inference(self, image_pil: Image.Image) -> Tuple[np.ndarray, List[dict]]:
        """
        Runs the MaskFormer model in panoptic mode and returns (seg_map, segments_info).

        Args:
            image_pil (PIL.Image): The input image in PIL format.

        Returns:
            seg_map (np.ndarray): A 2D array of shape (H, W) with segment IDs.
            segments_info (List[dict]): Metadata about each segment.
        """
        inputs = self.processor(images=image_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        result = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(image_pil.height, image_pil.width)],
            mask_threshold=self.threshold,
            overlap_mask_area_threshold=self.mask_area_threshold
        )[0]

        seg_map = result["segmentation"].numpy()
        segments_info = result["segments_info"]
        return seg_map, segments_info

    def _build_masks(self, seg_map: np.ndarray, segments_info: List[dict]) -> torch.Tensor:
        """
        Builds a tensor of Boolean masks from the segmentation map and segment info.

        Args:
            seg_map (np.ndarray): 2D array (H, W) with unique integer IDs per segment.
            segments_info (List[dict]): Segment metadata from MaskFormer.

        Returns:
            masks (torch.Tensor): A stack [N, H, W] of Boolean masks, one per segment.
        """
        all_masks = []
        for seg_info in segments_info:
            mask = (seg_map == seg_info["id"]).astype(np.uint8)
            if self.remove_small:
                mask, _ = self._remove_small_regions(
                    mask,
                    area_thresh=self.area_thresh,
                    mode=self.remove_mode
                )
            # Convert to torch BoolTensor
            all_masks.append(torch.as_tensor(mask, dtype=torch.bool))

        if not all_masks:
            return torch.empty(0)

        masks = torch.stack(all_masks, dim=0)
        return masks
