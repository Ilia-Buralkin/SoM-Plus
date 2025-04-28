import os
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import re
from ast import literal_eval
from .prompts import create_gqa_prompt_som_with_metadata, create_gqa_prompt_som_baseline, create_gqa_prompt_baseline


class GeminiGQA:
    """
    A simple class that queries Gemini for visual grounding in a RefCOCO-like scenario.
    Given a prompt and a labeled PIL image, it returns a single annotation ID from the model's response.
    """

    def __init__(self, model_name="gemini-1.5-pro", max_tokens=300, temperature=0, api_key=None):
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

    def get_answer(self, prompt: str, pil_image: Image.Image) -> tuple[str, list[str], str]:
        """
        Sends a single image + prompt to Gemini and parses out:
            - Final Answer: "<answer>"
            - Final Answer Variations: ['a', 'b', 'c', ...]

        Returns:
            final_answer (str): The quoted answer (1–2 words) after Final Answer:
            variations (List[str]): List of synonyms/alternatives
            text_response (str): Full raw response for logging/debugging
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

            # Relaxed: grab anything in quotes after Final Answer:
            answer_match = re.search(r'Final Answer:\s*["“](.+?)["”]', text_response, re.IGNORECASE)
            final_answer = answer_match.group(1).strip() if answer_match else "not_found"

            # Parse the literal list from Final Answer Variations:
            try:
                variations_match = re.search(r'Final Answer Variations:\s*(\[[^\]]*\])', text_response, re.IGNORECASE)
                #print(variations_match)
                variations = literal_eval(variations_match.group(1)) if variations_match else []
            except Exception as ve:
                print(f"Could not parse variations: {ve}")
                variations = []

            return final_answer, variations, text_response

        except Exception as e:
            print(f"Error during model call or parsing: {e}")
            return "error", [], str(e) 


class VertexGQA:
    """
    Rate-limited Vertex AI Gemini client for GQA tasks.
    """
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        max_tokens: int = 300,
        temperature: float = 0.0,
        qps: float = 1.0
    ):
        self.model = generative_models.GenerativeModel(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_interval = 1.0 / qps if qps > 0 else 0.0
        self._last_call = time.monotonic()
        self._lock = threading.Lock()

    def _throttle(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            wait = self.min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _encode_image(self, pil_image: Image.Image) -> bytes:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        return buf.getvalue()

    def get_answer(self, prompt: str, pil_image: Image.Image) -> tuple[str, list[str], str]:
        """
        Sends image+prompt to Vertex AI Gemini and returns final answer, variations, full text.
        """
        self._throttle()
        img_bytes = self._encode_image(pil_image)
        image_part = Part.from_data(data=img_bytes, mime_type="image/jpeg")
        text_part = Part.from_text(prompt)

        response = self.model.generate_content(
            [image_part, text_part],
            generation_config=generative_models.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=42
            )
        )
        text_response = response.text

        # Parse final answer
        ans_match = re.search(r'Final Answer:\s*"(.+?)"', text_response)
        final_answer = ans_match.group(1).strip() if ans_match else "not_found"

        # Parse variations
        vars_list = []
        var_match = re.search(r'Final Answer Variations:\s*(\[[^\]]*\])', text_response)
        if var_match:
            try:
                vars_list = literal_eval(var_match.group(1))
            except:
                pass
        # Ensure final is in list
        if final_answer not in vars_list:
            vars_list.append(final_answer)

        return final_answer, vars_list, text_response




def perform_gqa(question, image, descriptions, platform=None, model="gemini-1.5-pro", mode='parallel'):
    
    if platform == "Vertex": 
        vlm = VertexGQA(model_name=model)
    else: 
        vlm = GeminiGQA(model_name = model)

    if mode == "baseline":
        prompt = create_gqa_prompt_baseline(question)
    elif mode == "som_baseline":
        prompt = create_gqa_prompt_som_baseline(question)
    elif mode == "parallel" or mode == "unified":
        prompt = create_gqa_prompt_som_with_metadata(question, descriptions, desc_mode = mode)
    else: 
        print("Pick from : [baseline, som_baseline, parallel, unified]")

    final_answer, variations, text_response = vlm.get_answer(prompt, image)

    return final_answer, variations, text_response




