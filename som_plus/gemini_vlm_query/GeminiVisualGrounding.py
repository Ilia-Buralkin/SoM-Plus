from logging import raiseExceptions
import os
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import re
from .prompts import generate_prompt_baseline, generate_prompt_parallel_descriptions, generate_prompt_unified_descriptions, generate_prompt_w_descriptions_llm

class GeminiVisualGrounding:
    """
    A simple class that queries Gemini for visual grounding in a RefCOCO-like scenario.
    Given a prompt and a labeled PIL image, it returns a single annotation ID from the model's response.
    """

    def __init__(self, model_name="gemini-2.0-flash", max_tokens=200, temperature=0, api_key=None):
        """
        Args:
            model_name (str): Name/version of the Gemini model to use (e.g., 'gemini-2.0', 'gemini-2.0-flash', etc.).
            max_tokens (int): Maximum tokens in the model's response.
            api_key (str): Your Gemini (PaLM) API key. If not provided, must be set in the environment.
        """
        # If user gave explicit api_key, override; otherwise rely on environment
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        
        self.model = genai.GenerativeModel(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def encode_pil_image_to_base64(self, pil_image: Image.Image) -> str:
        """
        Converts a PIL Image to a base64 string with a data URI prefix.
        """
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_image}"

    def get_annotation_id(self, prompt: str, pil_image: Image.Image) -> int:
        """
        Sends a single image + prompt to Gemini and parses out 'Annotation_ID_x' from the response.

        Returns:
            annotation_id (int): The integer ID if found, else None.
        """
        encoded_image = self.encode_pil_image_to_base64(pil_image)
        try:
            response = self.model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"mime_type": "image/jpeg", "data": base64.b64decode(encoded_image.split(",")[1])},
                            {"text": prompt}
                        ]
                    }
                ],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens, 
                    temperature=self.temperature
                    )
            )
            text_response = response.text
            # Parse out "Annotation_ID_{number}" using a regex
            match = re.search(r"Annotation_ID_(\d+)", text_response)
            if match:
                return int(match.group(1))
            return None
        except Exception as e:
            print(f"Gemini error: {e}")
            return None
        
    def get_annotation_id_llm(self, prompt: str) -> int:
        """
        Sends a single image + prompt to Gemini and parses out 'Annotation_ID_x' from the response.

        Returns:
            annotation_id (int): The integer ID if found, else None.
        """
        try:
            response = self.model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens, 
                    temperature=self.temperature
                    )
            )
            text_response = response.text
            # Parse out "Annotation_ID_{number}" using a regex
            match = re.search(r"Annotation_ID_(\d+)", text_response)
            if match:
                return int(match.group(1))
            return None
        except Exception as e:
            print(f"Gemini error: {e}")
            return None


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



def perform_visual_grounding(question, image, descriptions, platform=None, model='gemini-1.5-pro', mode="parallel", vlm = True):

    if platform == "Vertex":
        grounding_vlm = VertexVisualGrounding(model_name=model)
    else:
        grounding_vlm = GeminiVisualGrounding(model_name=model)


    if vlm:
        if mode == "parallel":
            prompt = generate_prompt_parallel_descriptions(question, descriptions)
        elif mode == "unified":
            prompt = generate_prompt_unified_descriptions(question, descriptions)
        elif mode == "baseline":
            prompt = generate_prompt_baseline(question)
        else:
            print("Please pick the mode from [parallel, unified, baseline]")
        
        annotation_id = grounding_vlm.get_annotation_id(prompt, image)

    else:
        prompt = generate_prompt_w_descriptions_llm(query_sentences=question, annotation_descriptions=descriptions, desc_mode = mode)
        annotation_id = grounding_vlm.get_annotation_id_llm(prompt)

    return annotation_id, prompt

