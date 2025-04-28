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

warnings.simplefilter(action='ignore', category=FutureWarning)

env_path = "/home/iliab/rice/final_project/.env"
load_dotenv(dotenv_path=env_path)

# Import utility functions - keep the same as original
sys.path.append("/home/iliab/rice/final_project/clean_notebooks/packages_new")
from refcocog_testing_utils import (
    generate_prompt_baseline,
    generate_prompt_parallel_descriptions,
    generate_prompt_unified_descriptions,
    rescale_bounding_box,
    calculate_iou,
    is_center_inside,
    flatten_to_coords,
    rescale_polygon,
    process_segmentation,
    calculate_iou_with_multipolygons
)


import io
import time
import threading
import traceback
import re
from PIL import Image
from vertexai.generative_models import Part
from vertexai.preview import generative_models
import vertexai

class VertexVisualGrounding:
    """
    A class that queries Gemini models via Vertex AI for visual grounding in a RefCOCO-like scenario.
    Given a prompt and a labeled PIL image, it returns a single annotation ID from the model's response.

    Supports rate-limiting to respect a target QPS (queries per second).
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        max_tokens: int = 200,
        temperature: float = 0.0,
        project_id: str = None,
        location: str = None,
        qps: float = 1.0,
    ):
        # Initialize Vertex AI with project and location
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or "us-central1"
        vertexai.init(project=self.project_id, location=self.location)

        # Map external names to Vertex AI model identifiers
        model_mapping = {
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-2.0-pro": "gemini-2.0-pro",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-2.5-pro-preview-03-25": "gemini-2.5-pro-preview-03-25",
        }
        self.model_name = model_mapping.get(model_name, model_name)
        self.model = generative_models.GenerativeModel(self.model_name)

        # Generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Rate-limit configuration
        self.min_interval = 1.0 / qps
        self._last_call = 0.0
        self._lock = threading.Lock()

    def _throttle(self):
        """Sleep to enforce minimum interval between requests."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            wait = self.min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def get_annotation_id(self, prompt: str, pil_image: Image.Image) -> int:
        """
        Sends an image + prompt to Gemini via Vertex AI and returns the parsed Annotation_ID.

        Returns:
            annotation_id (int): The integer ID if found, else None.
        """
        try:
            # Throttle to respect QPS
            self._throttle()

            # Convert PIL image to bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()

            # Build request parts
            image_part = Part.from_data(data=img_bytes, mime_type="image/jpeg")
            text_part = Part.from_text(prompt)

            # Call the model
            response = self.model.generate_content(
                [image_part, text_part],
                generation_config=generative_models.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=1.0,
                    top_k=1,
                    seed=43,
                ),
            )

            # Parse the Annotation_ID
            text_response = response.text
            match = re.search(r"Annotation_ID_(\d+)", text_response)
            return int(match.group(1)) if match else None

        except Exception as e:
            print(f"VertexAI error: {e}")
            print(traceback.format_exc())
            return None

    def get_annotation_id_llm(self, prompt: str) -> int:
        """
        Sends a text-only prompt to Gemini via Vertex AI and returns the parsed Annotation_ID.
        """
        try:
            self._throttle()
            text_part = Part.from_text(prompt)
            response = self.model.generate_content(
                [text_part],
                generation_config=generative_models.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    seed=43,
                ),
            )
            text_response = response.text
            match = re.search(r"Annotation_ID_(\d+)", text_response)
            return int(match.group(1)) if match else None

        except Exception as e:
            print(f"VertexAI error: {e}")
            print(traceback.format_exc())
            return None



def generate_prompt_parallel_descriptions_llm(query_sentences, annotation_descriptions):
    """
    Generate a prompt to resolve a referring expression using only parallel descriptions.
    Keeping this function identical to the original.

    Args:
        query_sentences (list[str]): A list of referring expression sentences.
        annotation_descriptions (list[str]): A list of text descriptions for each annotation.

    Returns:
        str: Formatted prompt asking the model to select the best-matching annotation ID.
    """
    query_str = " ".join(query_sentences).strip()
    descriptions_block = "\n".join(annotation_descriptions)

    prompt = f"""
You are a reasoning assistant helping to resolve a user's query using only natural language descriptions of objects in an image. You do not have access to the image itself.

Below is a list of annotations and their associated descriptions:

{descriptions_block}

The user's query: "{query_str}"

Carefully compare the user's query with the descriptions. Pick exactly one annotation ID that corresponds best to this query.

Only output a single line in this format:
Annotation_ID_x

Do not include any commentary, justification, or explanation.
"""

    return textwrap.dedent(prompt).strip()


def test_som_combined(data, labeled_data_path, grounding_vlm, aug=False, single=False, iou_threshold=0.5, vlm=True):
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

        if single:
            with open(labeled_data_path + image_id + "_pro_single.txt", "r") as f:
                descriptions = f.read()
            boxes = [list(i['bbox']) for i in labels]
            masks = [i['segmentation'] for i in labels]
        else:
            descriptions = [i['description'] for i in labels]
            boxes = [list(i['bbox']) for i in labels]
            masks = [i['segmentation'] for i in labels]
        
        if vlm:
            prompt = generate_prompt_unified_descriptions(sentences, descriptions) if aug and single else \
                    generate_prompt_parallel_descriptions(sentences, descriptions) if aug else \
                    generate_prompt_baseline(sentences)
            output_id = grounding_vlm.get_annotation_id(prompt, labeled_image)
        else:
            prompt = generate_prompt_parallel_descriptions_llm(sentences, descriptions)
            output_id = grounding_vlm.get_annotation_id_llm(prompt)

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


def get_overall_results(df, grounding_model, segmentation_model, method):
    """
    Calculate overall results from the DataFrame.
    """
    # Box accuracy
    print(f"\nResults for {grounding_model} + {segmentation_model} + {method}")
    box_accuracy = df['center_match'].mean()
    print("Box Accuracy (center match):", box_accuracy)

    # IoU accuracy
    iou_accuracy = df['box_iou'].gt(0.5).mean()
    print("Box Accuracy (IoU > 0.5):", iou_accuracy)

    # Mask accuracy
    mask_accuracy = df['mask_iou'].gt(0.5).mean()
    print("Mask Accuracy (IoU > 0.5):", mask_accuracy)

    return box_accuracy, iou_accuracy, mask_accuracy


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
    parser.add_argument("--vlm", type=bool, default=True, 
                        help="Use VLM (True) or LLM (False)")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/iliab/rice/final_project/clean_notebooks/clean_resutls/vertex/refcocog/", 
                        help="Output directory")
    parser.add_argument("--qps", type=float, default=1.0, help="Adjust the rate for Gemini api (1.5 Pro)")
    args = parser.parse_args()
    
    # Set up paths based on your original structure
    if args.segmentation == "maskformer":
        labeled_data_path = "/home/iliab/rice/final_project/refcocog_dataset/selected_data/labeled_images_v4/maskformer/"
    else:  # sam2
        labeled_data_path = "/home/iliab/rice/final_project/refcocog_dataset/selected_data/labeled_images_v4/sam2/"
    
    test_data_path = "/home/iliab/rice/final_project/refcocog_dataset/selected_data/val_images_250/test_data.json"
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Initialize grounder with VertexAI
    grounder = VertexVisualGrounding(
        model_name=args.model,
        max_tokens=300,
        qps=args.qps
    )
    
    # Set up method parameters
    aug = args.method in ["parallel", "unified"]
    single = args.method == "unified"
    
    # Run test
    results = test_som_combined(
        data=test_data[:],
        labeled_data_path=labeled_data_path,
        grounding_vlm=grounder,
        aug=aug,
        single=single,
        iou_threshold=0.5,
        vlm=args.vlm
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_str = args.model.replace('-', '_')
    seg_str = args.segmentation
    method_str = args.method
    vlm_str = "vlm" if args.vlm else "llm"
    
    filename = f"vertex_results_{model_str}_{seg_str}_{method_str}_{vlm_str}.csv"
    results.to_csv(os.path.join(args.output_dir, filename), index=False)
    
    return results


if __name__ == "__main__":
    main()
