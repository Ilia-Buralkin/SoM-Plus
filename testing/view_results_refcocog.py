import pandas as pd
import re
import argparse
import os

def extract_info_variables(filename):
    """
    Extracts information from a filename and returns individual variables.

    Args:
        filename (str): The filename string.

    Returns:
        tuple: A tuple containing (model_name, segmentation_model, desc_mode, engine)
               or None if the filename doesn't match the pattern.
    """
    pattern = r"results_(gemini_1.5_pro|gemini_2.0_flash)_(sam2|maskformer)_(unified|parallel|baseline)_(vlm|llm).csv"
    print(filename)
    match = re.match(pattern, filename)
    if match:
        model_name = match.group(1)
        segmentation_model = match.group(2)
        desc_mode = match.group(3)
        engine = match.group(4)
        return model_name, segmentation_model, desc_mode, engine
    else:
        print("didnt find the match")

def print_results(data_path):
    """
    Calculate overall results from the DataFrame.
    """
    model, segmentation_model, desc_mode, engine = extract_info_variables(os.path.basename(data_path))
    df = pd.read_csv(data_path)
    df_shape = df.shape
    df_mask_accuracy_nonna = df['mask_iou'].notna().sum()
    iou_threshold = 0.5
    # Box accuracy
    print(f"\nResults for || {desc_mode} || {model} || {segmentation_model} || {engine}")
    print(f"Df shape {df_shape} || Mask accuracy notna {df_mask_accuracy_nonna}")

    box_accuracy = df['center_match'].mean()
    print("Box Accuracy (center match):", box_accuracy)

    # IoU accuracy
    iou_accuracy = df.loc[df['box_iou']>iou_threshold].shape[0]/len(df)
    print("Box Accuracy (IoU > 0.5):", iou_accuracy)

    # Mask accuracy
    mask_accuracy = df.loc[df['mask_iou']>iou_threshold].shape[0]/len(df)
    print("Mask Accuracy (IoU > 0.5):", mask_accuracy)

    return model, segmentation_model, desc_mode, engine, iou_accuracy, mask_accuracy

def process_directory(directory_path):
    """
    Iterates over files in a directory, extracts information using
    extract_info_variables, and prints the results.

    Args:
        directory_path (str): The path to the directory containing the files.
    """
    for filename in os.listdir(directory_path):
        print_results(directory_path+"/"+filename)

    return print("Done")

def main(): 
   parser =  argparse.ArgumentParser()
   parser.add_argument("--data_dir", type=str, default="flash/llm")
   args = parser.parse_args()
   base_path = "/home/iliab/rice/final_project/clean_notebooks/clean_resutls/final/refcocog/"
   full_path = base_path + args.data_dir 
   process_directory(full_path)


if __name__ == "__main__":
    main()
