import json
import os
import sys
import base64
import io
import pandas as pd
from PIL import Image
import google.generativeai as genai
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import argparse
from ast import literal_eval
import nltk
from nltk.corpus import wordnet as wn
from dotenv import load_dotenv
import textwrap
import vertexai
from vertexai.generative_models import Part
from vertexai.preview import generative_models
import time
import threading


# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load environment variables
load_dotenv()

# Download WordNet for WUPS calculation
nltk.download('wordnet', quiet=True)

try:
    from FlagEmbedding import BGEM3FlagModel
    embedding_model_available = True
except ImportError:
    embedding_model_available = False
    print("FlagEmbedding not installed. Embedding similarity will not be calculated.")


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

def create_gqa_prompt_baseline(question: str) -> str:
    """
    Creates a baseline prompt for GQA without SoM annotations.
    """
    return f"""
You are a visual reasoning assistant that helps in answering questions about images.

Question: {question}

Instructions:
1. Begin with **Reasoning:** and describe what you see and how you decide.
2. After your reasoning, on a new line write **only**:
   Final Answer: "answer"
3. Then write **5** possible variations of the final answer:
   - The **first** must be the plural or singular form of the original answer (e.g., 'dog' → 'dogs', or 'women' → 'woman').
   - The next **four** must be synonyms or semantically equivalent terms (e.g., 'woman' → 'lady', 'female', etc.).
   Final Answer Variations: ['variation1', 'variation2', 'variation3', 'variation4', 'variation5']
4. Use only lowercase, quoted strings.
5. Do **not** include any extra text, punctuation, capital letters, or explanation outside the specified format.
6. The answer inside the quotes should typically be one word, but at most two words if necessary. It must be placed in quotes "".
7. Carefully consider if the question requires just a yes or no answer.

Let's think step by step.
"""


def create_gqa_prompt_som_baseline(question: str) -> str:
    """
    Creates a prompt for GQA with SoM annotations.
    """
    return f"""
You are a visual reasoning assistant.  
The image is annotated in Set‑of‑Marks style: objects in the scene are labeled with numbers.

Question: {question}

Instructions:
1. Begin with **Reasoning:** and walk through your steps, referring to objects by their numbers (e.g. "Object 3 is a curtain…").
2. After your reasoning, on a new line write **only**:
   Final Answer: "answer"
3. Then write **5** variations of the answer:
   - The first must be the plural or singular form (opposite of the answer).
   - The remaining four must be synonyms or semantically equivalent terms.
   Final Answer Variations: ['variation1', 'variation2', 'variation3', 'variation4', 'variation5']
4. Use only lowercase, quoted strings.
5. Do **not** include any extra text, punctuation, or explanation outside the format.
6. The answer inside the quotes should typically be one word, but at most two words if necessary.
7. Carefully consider if the question requires just a yes or no answer.

Let's think step by step.
"""


def create_gqa_prompt_som_with_metadata(question: str, annotations_metadata: str, desc_mode: str) -> str:
    """
    Creates a prompt for GQA with SoM annotations and semantic metadata.
    
    Args:
        question (str): The question to answer
        annotations_metadata (str or list): Semantic descriptions
        desc_mode (str): Format of the annotations ("unified" or "parallel")
        
    Returns:
        str: Formatted prompt
    """
    if desc_mode == "unified":
        metadata_text = annotations_metadata
    elif desc_mode == "parallel":
        metadata_text = "\n".join(annotations_metadata)

    return f"""
You are a visual reasoning assistant.  
The image is annotated in the Set‑of‑Marks (SoM) style.  
Below is the detailed metadata for each numbered annotation—use this information to inform and focus your reasoning:

{metadata_text}

Question: {question}

Instructions:
1. Begin with **Reasoning:** and think step by step, referring naturally to annotation IDs (e.g. "Annotation_ID_1 shows…").
2. Use the provided metadata (description, color, location, etc.) to help inform your answer.
3. After your reasoning, write:
   Final Answer: "answer"
4. Then write 5 variations of the answer:
   - The first must be the plural or singular form (opposite of the original).
   - The next four must be synonyms or semantically equivalent terms.
   Final Answer Variations: ['variation1', 'variation2', 'variation3', 'variation4', 'variation5']
5. Use lowercase, quoted strings only.
6. Do not include any extra commentary or formatting.
7. The answer inside the quotes should typically be one word, but at most two words if necessary.
8. Carefully consider if the question requires just a yes or no answer.

Let's begin.
"""


def wups_score(pred, gt):
    """
    Calculate WUP similarity between prediction and ground truth.
    
    Args:
        pred (str): Predicted answer
        gt (str): Ground truth answer
        
    Returns:
        float: WUP similarity score (0.0-1.0)
    """
    # Convert to synsets (use first sense as heuristic)
    pred_syn = wn.synsets(pred)
    gt_syn = wn.synsets(gt)
    
    if not pred_syn or not gt_syn:
        return 0.0  # if no synset found

    # Compute WUPS for the first sense (can average all pairs too)
    return wn.wup_similarity(pred_syn[0], gt_syn[0]) or 0.0


def clean_text(text):
    """Clean and normalize text for comparison."""
    return text.strip().lower()


def compute_similarity(pred, gt, model):
    """
    Compute semantic similarity using embeddings.
    
    Args:
        pred (str): Predicted answer
        gt (str): Ground truth answer
        model: Embedding model
        
    Returns:
        float: Similarity score (0.0-1.0)
    """
    pred = clean_text(pred)
    gt = clean_text(gt)
    
    # Generate dense embeddings
    emb_pred = model.encode([pred])['dense_vecs']  # shape: (1, 1024)
    emb_gt = model.encode([gt])['dense_vecs']

    # Cosine similarity between normalized vectors
    emb_pred = F.normalize(torch.tensor(emb_pred), p=2, dim=1)
    emb_gt = F.normalize(torch.tensor(emb_gt), p=2, dim=1)
    sim = torch.matmul(emb_pred, emb_gt.T).item()

    return sim


def test_gqa_baseline(df, image_dir, vlm, som=False, desc_mode=None):
    """
    Test GQA performance with various prompt types.
    
    Args:
        df (DataFrame): Test data
        image_dir (str): Directory containing images
        vlm (GeminiGQA): VLM model instance
        som (bool): Whether to use SoM annotations
        desc_mode (str): Type of description mode if som=True
    
    Returns:
        DataFrame: Results with metrics
    """
    df = df.copy()
    df['final_answer'] = None
    df['full_response'] = None
    df['wups_score'] = None
    df['embedding_score'] = None
    df['strict_correct'] = None
    df['soft_correct'] = None
    df['final_answer_variations'] = None

    # Load embedding model if available
    if embedding_model_available:
        embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    else:
        embedding_model = None

    for i, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row['image_name']
        image_path = os.path.join(image_dir, image_name)

        try: 
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        question = row['query']

        if som:
            if desc_mode == "baseline":
                prompt = create_gqa_prompt_som_baseline(question)
            elif desc_mode == "unified_pro":
                prompt = create_gqa_prompt_som_with_metadata(question, row['single_description_pro'], "unified")
            elif desc_mode == "unified_flash":
                prompt = create_gqa_prompt_som_with_metadata(question, row['single_description_flash'], "unified")
            elif desc_mode == "parallel":
                prompt = create_gqa_prompt_som_with_metadata(question, row['multiple_descriptions'], "parallel")
        else:
            prompt = create_gqa_prompt_baseline(question)
        
        final_answer, variations, full_response = vlm.get_answer(prompt, image)
        
        # Ensure final answer is also in the variations
        if final_answer not in variations:
            variations.append(final_answer)
        
        # Evaluate correctness
        correct = row['answer'].strip().casefold() == final_answer.strip().casefold()
        soft_correct = any(
            row['answer'].strip().casefold() == v.strip().casefold()
            for v in variations
        ) 
        
        # Calculate WUPS score
        wups = wups_score(row['answer'], final_answer)
        
        # Calculate embedding similarity if model is available
        if embedding_model is not None:
            embedding = compute_similarity(row['answer'], final_answer, embedding_model)
        else:
            embedding = None

        # Store results
        df.at[i, "final_answer"] = final_answer
        df.at[i, "final_answer_variations"] = variations
        df.at[i, "full_response"] = full_response
        df.at[i, "strict_correct"] = correct
        df.at[i, "soft_correct"] = soft_correct
        df.at[i, "wups_score"] = wups
        df.at[i, "embedding_score"] = embedding
    
    # Print metrics
    print("Overall accuracy:", df['strict_correct'].mean())
    print("Overall soft accuracy:", df['soft_correct'].mean())
    print("Overall WUPS score:", df['wups_score'].mean())
    if embedding_model is not None:
        print("Overall embedding score:", df['embedding_score'].mean())

    return df


def main():
    parser = argparse.ArgumentParser(description="Test GQA with SoM and SoM+")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", 
                        choices=["gemini-1.5-pro", "gemini-2.0-flash"],
                        help="Model to use")
    parser.add_argument("--segmentation", type=str, default="maskformer", 
                        choices=["maskformer", "sam2", "none"],
                        help="Segmentation model (none for baseline without SoM)")
    parser.add_argument("--method", type=str, default="baseline", 
                        choices=["baseline", "parallel", "unified_pro", "unified_flash"],
                        help="SoM method to use")
    parser.add_argument("--output_dir", type=str, default="/home/iliab/rice/final_project/clean_notebooks/clean_resutls/vertex/gqa/", 
                        help="Output directory")
    parser.add_argument("--qps", type=float, default=1.0, help="Adjust qps for gemini 1.5 pro - 0.1")
    args = parser.parse_args()
    
    # Set up paths
    if args.segmentation == "none":
        # Original images without SoM annotations
        image_dir = "/home/iliab/rice/final_project/gqa_dataset/images/"
        test_data_path = "/home/iliab/rice/final_project/gqa_dataset/labeled_images_v3/maskformer/test_df.csv"
        som = False
        desc_mode = None
    else:
        # Images with SoM annotations
        image_dir = f"/home/iliab/rice/final_project/gqa_dataset/labeled_images_v3/{args.segmentation}/"
        test_data_path = f"/home/iliab/rice/final_project/gqa_dataset/labeled_images_v3/{args.segmentation}/test_df.csv"
        som = True
        desc_mode = args.method
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Initialize model
    vlm = VertexGQA(
        model_name=args.model,
        max_tokens=300,
        qps=args.qps
    )
    
    # Run test
    results = test_gqa_baseline(
        df=test_df,
        image_dir=image_dir,
        vlm=vlm,
        som=som,
        desc_mode=desc_mode
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    segmentation_str = args.segmentation if args.segmentation != "none" else "baseline"
    method_str = args.method if som else "no_som"
    filename = f"results_{args.model.replace('-', '_')}_{segmentation_str}_{method_str}.csv"
    results.to_csv(os.path.join(args.output_dir, filename), index=False)
    
    # Calculate and display summary metrics
    print(f"\nSummary for {args.model} with {segmentation_str} using {method_str} method:")
    print(f"Strict Accuracy: {results['strict_correct'].mean()*100:.1f}%")
    print(f"Soft Accuracy: {results['soft_correct'].mean()*100:.1f}%")
    print(f"Average WUPS Score: {results['wups_score'].mean():.3f}")
    if embedding_model_available:
        print(f"Average Embedding Similarity: {results['embedding_score'].mean():.3f}")
    
    return results


def process_dataframe_with_wups(df_path):
    """
    Process an existing results dataframe to update it with WUPS scores and correctness.
    
    Args:
        df_path (str): Path to the CSV file
    
    Returns:
        DataFrame: Updated DataFrame with additional metrics
    """
    df = pd.read_csv(df_path)
    
    # Initialize columns if they don't exist
    if 'wups_score' not in df.columns:
        df['wups_score'] = None
    
    if 'correct' not in df.columns:
        df['correct'] = None
    
    # Calculate WUPS scores
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(row['wups_score']):
            df.at[i, 'wups_score'] = wups_score(row['answer'], row['final_answer'])
        
        # Update correctness based on WUPS score
        if row['wups_score'] == 1.0:
            df.at[i, 'correct'] = True
    
    # Save updated DataFrame
    #df.to_csv(df_path.replace('.csv', '_updated.csv'), index=False)
    
    print(f"Original accuracy: {df['strict_correct'].mean()*100:.1f}%")
    print(f"With WUPS=1 correction: {df['correct'].mean()*100:.1f}%")
    
    return df


if __name__ == "__main__":
    main()
