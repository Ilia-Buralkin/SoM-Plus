
import os
import pandas as pd
import random
from datasets import load_dataset

ds = load_dataset("lmms-lab/GQA", "testdev_all_images")

queries = pd.read_csv("/home/iliab/rice/final_project/gqa_dataset/queries.csv")

image_ids = ds['testdev']['id']
sampled_ids = random.sample(image_ids, 300)

output_dir = "/home/iliab/rice/final_project/gqa_dataset/images"

# Create a lookup: id -> image
image_lookup = {item["id"]: item["image"] for item in ds['testdev'] if item["id"] in sampled_ids}

# Save the images
for image_id, image in image_lookup.items():
    image_path = os.path.join(output_dir, f"{image_id}.jpg")
    image.save(image_path)

print(f"Saved {len(image_lookup)} images to: {output_dir}")

queries['image_id'] = queries['image_name'].apply(lambda x: x.split(".")[0])
selected_queries = queries[queries["image_id"].isin(sampled_ids)]

# Optional: save to file
selected_queries.to_csv("/home/iliab/rice/final_project/gqa_dataset/selected_queries.csv", index=False)

