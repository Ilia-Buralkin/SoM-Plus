# segmentation_pipeline.py

import torch
import random
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class AbstractSegmentationPipeline(ABC):
    """
    Abstract base class for segmentation pipelines that provides a common interface
    and shared functionality for different segmentation models (MaskFormer, SAM2, etc.).
    """

    def __init__(
        self,
        threshold: float = 0.8,
        mask_area_threshold: float = 0.5,
        mask_fill_alpha: float = 0.3,
        mask_border_thickness: int = 2,
        remove_small: bool = False,
        area_thresh: int = 100,
        remove_mode: str = "islands",
    ):
        """
        Constructor for AbstractSegmentationPipeline.

        Args:
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
        self.threshold = threshold
        self.mask_area_threshold = mask_area_threshold
        self.mask_fill_alpha = mask_fill_alpha
        self.mask_border_thickness = mask_border_thickness
        self.remove_small = remove_small
        self.area_thresh = area_thresh
        self.remove_mode = remove_mode
        
        # Ensure remove_mode is valid
        assert remove_mode in ["holes", "islands"], "remove_mode must be 'holes' or 'islands'"

    @abstractmethod
    def segment_and_annotate(self, image: Image.Image):
        """
        Main entry point: 
        1) Runs model-specific segmentation on an input PIL image,
        2) Optionally processes the masks (removing small regions, etc.),
        3) Draws each mask + numeric label onto a copy of the image.

        Returns:
            annotated_image (PIL.Image):
                The input image with colored masks and numeric labels drawn.
            annotations (List[dict]):
                A list of annotation dicts for each detected mask, sorted by area.
        """
        pass

    @abstractmethod
    def _run_inference(self, image: Image.Image):
        """
        Runs the specific segmentation model on the input image.
        Implementation is model-specific.

        Returns:
            Model-specific output that will be further processed into masks.
        """
        pass

    def visualize_single_annotation(
        self,
        image: Image.Image,
        annotations: List[dict],
        label_index: int = 1,
        mask_thickness: int = 3,
        show_bbox: bool = False,
        blur_strength: int = 0,
        grayscale_background: bool = True,
        show_label: bool = True
    ) -> Image.Image:
        """
        Visualize a single mask annotation on a clean copy of the image with
        everything else blurred out or grayscaled.

        Args:
            image (PIL.Image):
                The input image to which you want to overlay one mask + label.
            annotations (List[dict]):
                List of annotation dicts from segment_and_annotate().
            label_index (int):
                Which annotation index to visualize (1-based).
            mask_thickness (int):
                Thickness of the mask border in pixels.
            show_bbox (bool):
                Whether to also draw the bounding box in addition to the mask + label.
            blur_strength (int):
                Strength of the blur effect for the background (higher values = more blur).
            grayscale_background (bool):
                Whether to also grayscale the background in addition to blurring it.

        Returns:
            overlay_image (PIL.Image):
                A new image with the selected annotation in color and everything else modified.
        """
        image_bgr = self._pil_to_bgr(image)
        original_bgr = image_bgr.copy()

        # Get the relevant annotation
        if label_index <= 0 or label_index > len(annotations):
            raise ValueError(f"Invalid label_index: {label_index}")
        ann = annotations[label_index - 1]

        # Convert boolean mask if needed
        mask = ann["segmentation"]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        if blur_strength > 0:
            # Create a blurred version of the entire image
            image_bgr = cv2.GaussianBlur(image_bgr, (blur_strength, blur_strength), 0)
        
        # If grayscale is requested, convert the blurred background to grayscale
        if grayscale_background:
            gray_1ch = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            image_bgr = cv2.cvtColor(gray_1ch, cv2.COLOR_GRAY2BGR)
        
        # Create the composite image: modified background + original foreground
        composite_bgr = image_bgr.copy()
        composite_bgr[mask == 1] = original_bgr[mask == 1]
        
        # Get a random color for the mask border
        color = self._random_bgr_color()

        # Draw just the border (not fill since we want to see the original content)
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(composite_bgr, contours, -1, color, thickness=mask_thickness)

        # Draw label in approximate mask centroid
        cX, cY = self._get_valid_centroid(mask, other_masks=[])
        font_scale = self._compute_font_scale_from_bbox(ann["bbox"])
        
        if show_label:
            self._draw_label(
                image=composite_bgr,
                text=str(label_index),
                position=(cX, cY),
                font_scale=font_scale
            )

        # Optionally draw bounding box
        if show_bbox:
            x, y, w, h = map(int, ann["bbox"])
            cv2.rectangle(composite_bgr, (x, y), (x + w, y + h), color, 2)

        overlay_image = self._bgr_to_pil(composite_bgr)
        return overlay_image

    def _generate_annotations(self, masks) -> List[dict]:
        """
        Converts masks to bounding boxes, computes area, etc.

        Args:
            masks: List or tensor of boolean masks.

        Returns:
            List[dict]: One dictionary per mask with fields like:
                {
                   "segmentation": <mask>,
                   "area": <float>,
                   "bbox": [x, y, w, h],
                   "predicted_iou": 1.0,
                   "point_coords": [0, 0],
                }
        """
        outputs = []
        
        # Handle both list and tensor input
        if isinstance(masks, torch.Tensor):
            masks_list = [masks[i].cpu().numpy() for i in range(masks.size(0))]
        else:
            masks_list = masks
            
        for mask in masks_list:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask
                
            # Compute bounding box
            if mask_np.sum() > 0:
                ys, xs = np.where(mask_np)
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            else:
                # Empty mask
                bbox = [0.0, 0.0, 0.0, 0.0]
                
            ann = {
                "segmentation": mask_np,
                "area": float(mask_np.sum()),
                "bbox": bbox,
                "predicted_iou": 1.0,      # placeholder
                "point_coords": [0.0, 0.0] # placeholder
            }
            outputs.append(ann)

        # Sort by area (descending) so the largest mask is first
        return sorted(outputs, key=lambda x: x["area"], reverse=True)

    def _draw_masks_and_labels(
        self,
        image_bgr: np.ndarray,
        masks,
        annotations: List[dict]
    ):
        """
        Draws each mask + label onto an existing BGR image array in-place.
        Handles both list and tensor input for masks.
        """
        used_masks = []
        label_boxes = []  # track bounding boxes of drawn labels to avoid overlap

        for i, ann in enumerate(annotations, start=1):
            mask = ann["segmentation"]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            color = self._random_bgr_color()
            self._overlay_mask_with_border(
                image=image_bgr,
                mask=mask,
                border_color=color,
                border_thickness=self.mask_border_thickness,
                fill_color=color,
                fill_alpha=self.mask_fill_alpha
            )

            # Find a centroid that doesn't overlap previously drawn labels
            cX, cY = self._get_valid_centroid(mask, used_masks)
            used_masks.append(mask)

            # Scale label size based on bounding box
            font_scale = self._compute_font_scale_from_bbox(ann["bbox"])
            label_str = str(i)

            # Attempt to place the label without overlapping previous labels
            (tw, th), base = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            for _ in range(20):
                proposed_tl = (int(cX - tw / 2), int(cY - th / 2 - base))
                proposed_br = (int(cX + tw / 2), int(cY + th / 2))

                # Check overlap with prior label boxes
                if all(not self._rects_overlap((proposed_tl, proposed_br), lb) for lb in label_boxes):
                    break
                cY += 5  # nudge downward and try again

            label_rect = self._draw_label(image_bgr, label_str, (cX, cY), font_scale=font_scale)
            label_boxes.append(label_rect)

    # -------------------------------------------------------------------------
    #                          Utility / Helper Methods
    # -------------------------------------------------------------------------

    def _remove_small_regions(self, mask: np.ndarray, area_thresh: float, mode: str):
        """
        Removes small disconnected 'islands' or fills small 'holes' in a binary mask.
        """
        assert mode in ["holes", "islands"], "mode must be 'holes' or 'islands'"

        correct_holes = (mode == "holes")
        # XOR the mask if we're filling holes
        working_mask = (correct_holes ^ mask).astype(np.uint8)

        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[1:, -1]  # skip background label

        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0:
            return mask, False

        fill_labels = [0] + small_regions
        if not correct_holes:
            # For islands mode, keep everything except the small ones
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep the largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]

        mask = np.isin(regions, fill_labels)
        return mask, True

    def _compute_font_scale_from_bbox(self, bbox: List[float]) -> float:
        """
        Scales the label font size based on bounding box height (for consistency).
        """
        _, _, _, h = bbox
        min_h, max_h = 10, 300
        min_scale, max_scale = 0.4, 0.9

        # Clamp the bounding box height
        h = max(min(h, max_h), min_h)
        norm = (h - min_h) / (max_h - min_h)
        return min_scale + norm * (max_scale - min_scale)

    def _get_valid_centroid(
        self,
        mask: np.ndarray,
        other_masks: List[np.ndarray],
        border_padding: int = 5
    ) -> Tuple[int, int]:
        """
        Finds a centroid for label placement that:
          - is inside 'mask'
          - does not overlap with any previously used mask
          - is not too close to the image border
        """
        h, w = mask.shape
        mask_u8 = mask.astype(np.uint8)

        combined_others = np.zeros((h, w), dtype=np.uint8)
        for om in other_masks:
            combined_others |= om.astype(np.uint8)

        valid_area = np.logical_and(mask_u8, ~combined_others).astype(np.uint8)

        # Remove a 'border_padding' region around the image edges
        valid_area[:border_padding, :] = 0
        valid_area[-border_padding:, :] = 0
        valid_area[:, :border_padding] = 0
        valid_area[:, -border_padding:] = 0

        if valid_area.sum() == 0:
            return self._compute_mask_centroid(mask)

        dist_transform = cv2.distanceTransform(valid_area, cv2.DIST_L2, 5)
        max_loc = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        # Numpy returns (row, col) -> (y, x)
        return int(max_loc[1]), int(max_loc[0])

    def _compute_mask_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        Fallback centroid if there's no 'valid area' for label placement.
        """
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
        else:
            # No foreground
            return 0, 0

    def _overlay_mask_with_border(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        border_color=(255, 0, 0),
        border_thickness=2,
        fill_color=(255, 0, 0),
        fill_alpha=0.3
    ):
        """
        Fills the 'mask' region on 'image' with fill_color (alpha blended)
        and draws a border in border_color.
        """
        overlay = image.copy()
        overlay[mask == 1] = (
            overlay[mask == 1] * (1.0 - fill_alpha)
            + np.array(fill_color, dtype=np.uint8) * fill_alpha
        ).astype(np.uint8)

        # Draw thick contours
        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, border_color, thickness=border_thickness)

        image[:] = overlay

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.75,
        text_color=(255, 255, 255),
        text_thickness=2,
        bg_color=(0, 0, 0),
        alpha: float = 0.6
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Draws label text with a semi-transparent background. Returns
        the rectangle (top-left, bottom-right) for overlap checking.
        """
        (tw, th), base = cv2.getTextSize(text, font, font_scale, text_thickness)
        cX, cY = position
        rect_tl = (int(cX - tw / 2), int(cY - th / 2 - base))
        rect_br = (int(cX + tw / 2), int(cY + th / 2))

        # Clip to image bounds
        rect_tl = (max(0, rect_tl[0]), max(0, rect_tl[1]))
        rect_br = (min(image.shape[1], rect_br[0]), min(image.shape[0], rect_br[1]))

        x1, y1 = rect_tl
        x2, y2 = rect_br
        roi = image[y1:y2, x1:x2].copy()

        if roi.size > 0:
            overlay = roi.copy()
            overlay[:] = bg_color
            blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
            image[y1:y2, x1:x2] = blended

        text_org = (int(cX - tw / 2), int(cY + th / 2))
        cv2.putText(image, text, text_org, font, font_scale, text_color, text_thickness, lineType=cv2.LINE_AA)
        return rect_tl, rect_br

    def _rects_overlap(self, rect1, rect2, padding=2):
        """
        Checks whether two rectangles (tl, br) overlap, with optional 'padding'.
        """
        (x1_min, y1_min), (x1_max, y1_max) = rect1
        (x2_min, y2_min), (x2_max, y2_max) = rect2

        x1_min -= padding; y1_min -= padding
        x1_max += padding; y1_max += padding
        x2_min -= padding; y2_min -= padding
        x2_max += padding; y2_max += padding

        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

    def _random_bgr_color(self) -> Tuple[int, int, int]:
        """Returns a random BGR color tuple."""
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    def _pil_to_bgr(self, image_pil: Image.Image) -> np.ndarray:
        """
        Converts a PIL (RGB) image to an OpenCV-compatible BGR numpy array.
        """
        image_rgb = np.asarray(image_pil).copy()
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def _bgr_to_pil(self, image_bgr: np.ndarray) -> Image.Image:
        """
        Converts a BGR numpy array (OpenCV style) to a PIL (RGB) image.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
