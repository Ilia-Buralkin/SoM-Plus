# SoM+ Package

A Python toolkit that combines the Set-of-Mark prompting with additional feature augmentations for each of the annotations. Segmentation is performed with MaskFormer/SAM2 and then augmented via Gemini 2.0 Flash. 

![Pipeline_overview](https://github.com/user-attachments/assets/03818d5b-c93f-4384-a04e-5f45a1a50af5)

Figure 1: The SoM+ pipeline with Parallel approach (top) and Unified approach (bottom). Both methods generate structured
semantic descriptions from segmented regions to enhance visual grounding

## Quantitative results

SoM+ Parallel improves visual grounding across both SAM 2 and MaskFormer segmentation methods when using Gemini 1.5 Pro. However, when using Gemini Flash 2.0 model, the benefit of semantic descriptions appears only in suboptimal segmentation settings (SAM 2 prompt based segmentation).

Metric: Referring Expression Comprehension (REC) Accuracy - bounding box IoU > 50 %

![maskformer_rec_accuracy](https://github.com/user-attachments/assets/0364109f-d184-446c-9484-eae6290893d3)

RefCOCOg results with Maskformer segmentation.

![sam2_rec_accuracy](https://github.com/user-attachments/assets/6f760c04-a686-4d5e-8a75-a8a56bf0aecf)

RefCOCOg results with SAM 2 Segmentation.

## Prerequisites

- **Python** 3.10  

## Installation

1. **Clone & Install _som_plus_**

   ```bash
   git clone https://github.com/<your-org>/som_plus.git
   cd som_plus
   pip install -e .

2. **Clone & Install SAM2**

   ```bash
   cd ..
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .

3. Download SAM2 checkpoint
   ```bash
   mkdir -p checkpoints
   wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt \
        -O checkpoints/sam2.1_hiera_base_plus.pt


## Demo 

To get a better idea, run the demo below. Do not forget to define GEMINI_API_KEY with a .env file to configure genai. 

demo.ipynb
