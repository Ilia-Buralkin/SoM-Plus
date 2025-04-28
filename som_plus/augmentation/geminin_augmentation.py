# gemini_annotator.py

import os
import json
import base64
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import google.generativeai as genai


class GeminiAugmentor:
    """
    A wrapper class for interacting with the Gemini model
    from the google.generativeai library.
    """

    def __init__(self, model_name='gemini-2.0-flash', max_tokens=200, api_key=None):
        """
        Constructor to set up the Gemini model.

        Args:
            model_name (str): The ID of the Gemini model variant.
            max_tokens (int): The maximum tokens in the response.
            api_key (str): If not None, we configure genai with this key here.
        """
        if api_key:
            genai.configure(api_key=api_key)
        else: 
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        self.model = genai.GenerativeModel(model_name)
        self.max_tokens = max_tokens

    def describe_annotation(self, prompt: str, encoded_image: str) -> str:
        """
        Send a single image+prompt pair to Gemini for text generation.

        Args:
            prompt (str): The text prompt for the Gemini model.
            encoded_image (str): A base64-encoded string of a JPEG image.
                                Format: "data:image/jpeg;base64,...."

        Returns:
            str: The generated text from Gemini, or None on error.
        """
        try:
            # Extract the part after "base64,"
            _, b64_part = encoded_image.split(",", 1)
            image_data = base64.b64decode(b64_part)
            response = self.model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"mime_type": "image/jpeg", "data": image_data},
                            {"text": prompt}
                        ]
                    }
                ],
                generation_config=genai.types.GenerationConfig(max_output_tokens=self.max_tokens)
            )
            return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            return None

    def parallel_describe(
        self,
        prompts: list,
        encoded_images: list,
        max_workers=5
    ) -> list:
        """
        In parallel, send multiple (prompt, encoded_image) pairs to Gemini.

        Args:
            prompts (list[str]): A list of text prompts.
            encoded_images (list[str]): A matching list of base64-encoded images.
            max_workers (int): Number of threads to use.

        Returns:
            list[str]: The list of Gemini responses, or "UNKNOWN" for failures.
        """
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self.describe_annotation, prompt, img): i
                for i, (prompt, img) in enumerate(zip(prompts, encoded_images))
            }
            for future in tqdm(as_completed(future_map), total=len(prompts)):
                idx = future_map[future]
                resp = future.result()
                results[idx] = resp if resp else "UNKNOWN"
        return results


# ---------------------------------------------------------------------
# Additional utility functions for annotation augmentation
# ---------------------------------------------------------------------

def encode_image_from_pil(image: Image.Image) -> str:
    """
    Convert a PIL image to a base64-encoded JPEG string:
    "data:image/jpeg;base64, <base64data>".
    """
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def create_single_mask_prompt(label_num: int) -> str:
    """
    Returns a specialized prompt for describing a single annotation/mask.

    For example, your prompt format:
    """
    prompt = f"""
Your task is to describe the specific object inside the provided annotation.

Important Note!: **In this image, the object of interest appears in color with a colored outline/mask, while everything outside the annotation is shown in grayscale. Focus on the colored object only! Even if it is a background**

⚠️ Do NOT describe:
- The annotation outline or color.
- Any objects outside the annotation mask.

Focus **only on the object that has the label number inside the mask**, and ensure your response is based on its visible appearance and attributes.
Be careful with large annotations that might be background. 

Use the exact structure below:

Annotation_ID_{label_num}:
    Description: A brief but detailed description of the object (e.g., what it is, what it's doing, etc.).
    Relative location: Where the object is in the image (e.g., location in the image, next to which other objects).
    Color of the object inside annotation: Primary visible colors.
    Texture / pattern: (e.g., striped, furry, smooth, patterned).
    Pose / action: What the object is doing or its orientation or how is it being used (e.g., grazing, sitting, held by a woman).
    Number of similar objects in scene: How many similar objects.
    Unique distinguishing features: Any unique attributes or details.
"""
    return prompt

def augment_annotations_parallel(
    pipeline,
    image: Image.Image,
    annotations: list,
    annotator = GeminiAugmentor(),
    max_workers=5,
    grayscale_background=True, 
    blur_strength=0,
    show_label=True
) -> list:
    """
    For each annotation in `annotations`, we:
      1) Render a single annotation mask on a fresh copy of the original image
      2) Base64-encode that cropped or masked image
      3) Send it to Gemini with an appropriate prompt
      4) Store the textual description in `ann['description']`
    Finally, returns a new list of annotations sorted by area descending.
    
    Args:
        pipeline: The segmentation pipeline object
        image: The original PIL image
        annotations: List of annotation dictionaries
        annotator: GeminiAnnotator object for generating descriptions
        max_workers: Number of parallel workers for API calls
        grayscale_background: Whether to grayscale the areas outside the mask
        blur_strength: Strength of blur to apply to background (0 for no blur)
        
    Returns:
        List of augmented annotations with descriptions, sorted by area
    """
    prompts = []
    encoded_images = []
    
    # For each annotation, create a small annotated image just for that mask
    for i, _ in enumerate(annotations):
        label_idx = i + 1
        single_img = pipeline.visualize_single_annotation(
            image=image, 
            annotations=annotations, 
            label_index=label_idx,
            grayscale_background=grayscale_background,
            blur_strength=blur_strength, 
            show_label=show_label
        )
        
        b64_img = encode_image_from_pil(single_img)
        p = create_single_mask_prompt(label_idx)
        prompts.append(p)
        encoded_images.append(b64_img)
    
    # Query Gemini in parallel
    results = annotator.parallel_describe(prompts, encoded_images, max_workers)
    
    # Attach descriptions to annotations
    augmented = []
    for ann, desc in zip(annotations, results):
        ann["description"] = desc
        augmented.append(ann)
    
    # Sort by area again (largest first)
    return sorted(augmented, key=lambda x: x["area"], reverse=True)


def augment_annotations_unified(
    annotated_image: Image.Image,
    annotations: list,
    image_id,
    annotator=GeminiAugmentor(model_name='gemini-1.5-pro', max_tokens=1000)
) -> str:
    """
    Single pass approach:
      1) annotated_image is an image with multiple labeled masks
      2) We generate a single prompt that covers all N annotations at once
      3) We get back a single text response with descriptions for each annotation

    Returns the raw text from Gemini.
    """
    # Base64 the entire annotated image
    b64_img = encode_image_from_pil(annotated_image)
    num_objs = len(annotations)

    # Single prompt describing all labeled masks
    prompt = f"""
You are given an image containing {num_objs} labeled object masks. Each object has a numeric label inside its mask.

⚠️ Describe only the object inside each mask (label #1, #2, etc.).
Do not mention the outlines or numbers themselves.

For each label, provide:

Annotation_ID_<label_number>:
    Description: A brief but detailed description of the object (e.g., what it is, what it's doing, etc.).
    Relative location: Where the object is in the image (e.g., location in the image, next to which other objects).
    Color of the object inside annotation: Primary visible colors.
    Texture / pattern: (e.g., striped, furry, smooth, patterned).
    Pose / action: What the object is doing or its orientation or how is it being used (e.g., grazing, sitting, held by a woman).
    Number of similar objects in scene: How many similar objects.
    Unique distinguishing features: Any unique attributes or details.

Return your output in this exact format, from label #1 to label #{num_objs}.
"""
    # Call Gemini
    response = annotator.describe_annotation(prompt, b64_img)
    if not response:
        print(f"Gemini failed for image {image_id}")
        raise ValueError(f"Stopping, inspect image {image_id}")
        
    return response

