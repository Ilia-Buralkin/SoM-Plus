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
    pattern = r"results_(gemini_1.5_pro|gemini_2.0_flash)_(sam2|maskformer)_(unified|parallel|baseline).csv"
    print(filename)
    match = re.match(pattern, filename)
    if match:
        model_name = match.group(1)
        segmentation_model = match.group(2)
        desc_mode = match.group(3)
        return model_name, segmentation_model, desc_mode
    else:
        print("didnt find the match")

def print_results(data_path):
    """
    Calculate overall results from the DataFrame.
    """
    model, segmentation_model, desc_mode = extract_info_variables(os.path.basename(data_path))
    df = pd.read_csv(data_path)
    mean_accuracy = df['strict_correct'].mean()
    mean_soft_accuracy = df['soft_correct'].mean()
    # Box accuracy
    print(f"\nResults for || {desc_mode} || {model} || {segmentation_model} || {engine}")
    print(f"Strict Accuracy = {mean_accuracy}")
    print(f"Soft Accuracy = {mean_soft_accuracy}")
    
    return None

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
   parser.add_argument("--data_dir", type=str, default="refcocog/flash/llm")
   args = parser.parse_args()
   base_path = "/home/iliab/rice/final_project/clean_notebooks/clean_resutls/final/"
   full_path = base_path + args.data_dir 
   process_directory(full_path)


if __name__ == "__main__":
    main()
