# Standard library imports
from logging import raiseExceptions
import textwrap  # Already included in your file


def generate_prompt_w_descriptions_llm(query_sentences, annotation_descriptions, desc_mode):
    """
    Args:
        query_sentences (list[str]): A list of referring expression sentences (e.g., ["zebra in the middle"]).
        annotation_descriptions (list[str]): A list of text descriptions for each annotation, 
                                             each presumably containing the structured fields 
                                             (Description, Relative location, etc.).
    
    Returns:
        A formatted prompt (string) that clearly presents all annotation descriptions 
        and asks the model to pick one annotation ID for the given query.
    """
    query_str = " ".join(query_sentences).strip()
    if desc_mode == 'parallel':
        descriptions_block = "\n".join(annotation_descriptions)
    elif desc_mode == "unified":
        descriptions_block = annotation_descriptions
    else: 
        raise ValueError("LLM prompt must include parallel or unified descriptions")

    prompt = f"""
You are a reasoning assistant helping to resolve a user’s query using only natural language descriptions of objects in an image. You do not have access to the image itself.

Below is a list of annotations and their associated descriptions:

{descriptions_block}

The user's query: "{query_str}"

Carefully compare the user’s query with the descriptions. Pick exactly one annotation ID that corresponds best to this query.

Only output a single line in this format:
Annotation_ID_x

Do not include any commentary, justification, or explanation.
"""

    # Use textwrap.dedent to remove extra indentation, then strip leading/trailing whitespace
    prompt = textwrap.dedent(prompt).strip()
    return prompt

def generate_prompt_baseline(sentences):
    prompt = f"""
    You are given an image with multiple annotated objects, each labeled with a numeric ID.
    The user's query: {sentences}
    Pick exactly one annotation ID that corresponds best to this query.
    Write it as: Annotation_ID_x
    Remember not to mention anything else.
    """
    return prompt

import textwrap

def generate_prompt_parallel_descriptions(query_sentences, annotation_descriptions):
    """
    Args:
        query_sentences (list[str]): A list of referring expression sentences (e.g., ["zebra in the middle"]).
        annotation_descriptions (list[str]): A list of text descriptions for each annotation, 
                                             each presumably containing the structured fields 
                                             (Description, Relative location, etc.).
    
    Returns:
        A formatted prompt (string) that clearly presents all annotation descriptions 
        and asks the model to pick one annotation ID for the given query.
    """

    # Join the user’s referring expression sentences (if there are multiple)
    query_str = " ".join(query_sentences).strip()

    # Build a readable multiline block of annotation descriptions, each on its own paragraph.
    # Since each description might already contain newline formatting, 
    # we'll just separate them with a blank line for clarity.
    descriptions_block = "\n\n".join(annotation_descriptions)

    # Create the final prompt
    prompt = f"""
You are given an image with multiple annotated objects, each labeled with a numeric ID. 
Below is a list of those object IDs and their descriptions:

{descriptions_block}

The user's query: "{query_str}"

Pick exactly one annotation ID that corresponds best to this query while utilizing the descriptions, and write it in this format:
Annotation_ID_x

Do not include any additional commentary. 
Only produce the line "Annotation_ID_x" and nothing else.
"""

    # Use textwrap.dedent to remove extra indentation, then strip leading/trailing whitespace
    prompt = textwrap.dedent(prompt).strip()
    return prompt

def generate_prompt_unified_descriptions(sentences, annotation_descriptions):
    prompt = f"""
    You are given an image with multiple annotated objects, each labeled with a numeric ID. 
Below is a list of those object IDs and their descriptions:

{annotation_descriptions}

The user's query: "{sentences}"

Pick exactly one annotation ID that corresponds best to this query while utilizing the descriptions, and write it in this format:
Annotation_ID_x

Do not include any additional commentary. 
Only produce the line "Annotation_ID_x" and nothing else.
    """
    return prompt


def create_gqa_prompt_baseline(question: str) -> str:
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

Let’s think step by step.

Final Answer: "answer"
Final Answer Variations: ['variation1', 'variation2', 'variation3', 'variation4', 'variation5']
"""


def create_gqa_prompt_som_baseline(question: str) -> str:
    return f"""
You are a visual reasoning assistant.  
The image is annotated in Set‑of‑Marks style: objects in the scene are labeled with numbers.

Question: {question}

Instructions:
1. Begin with **Reasoning:** and walk through your steps, referring to objects by their numbers (e.g. “Object 3 is a curtain…”).
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

Let’s think step by step.

Final Answer: "answer"
Final Answer Variations: ['variation1', 'variation2', 'variation3', 'variation4', 'variation5']
"""

def create_gqa_prompt_som_with_metadata(question: str, annotations_metadata: str, desc_mode: str) -> str:
    if desc_mode == "unified":
        annotations_metadata = annotations_metadata
    elif desc_mode == "parallel":
        annotations_metadata = "\n".join(annotations_metadata)

    return f"""
You are a visual reasoning assistant.  
The image is annotated in the Set‑of‑Marks (SoM) style.  
Below is the detailed metadata for each numbered annotation—use this information to inform and focus your reasoning:

{annotations_metadata}

Question: {question}

Instructions:
1. Begin with **Reasoning:** and think step by step, referring naturally to annotation IDs (e.g. “Annotation_ID_1 shows…”).
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

Let’s begin.

Final Answer: "answer"
Final Answer Variations: ['variation1', 'variation2', 'variation3', 'variation4', 'variation5']
"""

