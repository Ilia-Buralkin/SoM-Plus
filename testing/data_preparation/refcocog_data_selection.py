import os
import random
import json
import requests
import pickle

# Original metadata paths
og_questions_path = "/home/iliab/rice/final_project/refcocog_dataset/refcocog_refseg/orig/refcocog/refs(google).p"
og_js_path = "/home/iliab/rice/final_project/refcocog_dataset/refcocog_refseg/orig/refcocog/instances.json"

# Directory to save selected images
output_image_dir = "/home/iliab/rice/final_project/refcocog_dataset/selected_data/val_images_250"

os.makedirs(output_image_dir, exist_ok=True)

# Load original metadata
questions_answers = pickle.load(open(og_questions_path, "rb"))
images_w_annotations = json.load(open(og_js_path, "r"))


def download_image(url, image_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return True
    except requests.RequestException:
        pass
    return False

# Filter only val split questions
val_questions = [q for q in questions_answers if q['split'] == 'val']
random.seed(42)  # for reproducibility

matched_data = []
used_image_ids = set()
remaining_questions = val_questions.copy()

while len(matched_data) < 250 and remaining_questions:
    question = random.choice(remaining_questions)
    remaining_questions.remove(question)

    image_id = question['image_id']
    if image_id in used_image_ids:
        continue  # skip duplicate images

    image_info = next((img for img in images_w_annotations['images'] if img['id'] == image_id), None)
    if not image_info or not image_info.get("flickr_url"):
        continue

    url = image_info["flickr_url"]
    image_path = os.path.abspath(os.path.join(output_image_dir, f"{image_id}.jpg"))

    if download_image(url, image_path):
        annotations_info = [ann for ann in images_w_annotations['annotations'] if ann['image_id'] == image_id]

        # Add local path to image_info
        image_info = image_info.copy()  # avoid mutating original dataset
        image_info['local_path'] = image_path

        matched_data.append({
            "question": question,
            "image_info": image_info,
            "annotations_info": annotations_info
        })

        used_image_ids.add(image_id)
    else:
        print(f"Failed to download image {image_id} from {url}, trying another one...")

# Example output check
print(f"Collected and downloaded metadata for {len(matched_data)} unique val images.")
print(json.dumps(matched_data[:2], indent=2))


json.dump(matched_data, open("/home/iliab/rice/final_project/refcocog_dataset/selected_data/val_images_250/test_data.json", "w"), indent=2)
