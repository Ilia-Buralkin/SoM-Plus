# SoM+ Package

A Python toolkit that combines the Set-of-Mark prompting with additional feature augmentations for each of the annotations. Segmentation is performed with MaskFormer/SAM2 and then augmented via Gemini 2.0 Flash. 

[Pipeline_overview.pdf](https://github.com/user-attachments/files/19961195/Pipeline_overview.pdf)
Figure 1: The SoM+ pipeline with Parallel approach (top) and Unified approach (bottom). Both methods generate structured
semantic descriptions from segmented regions to enhance visual grounding

## Quantitative results

SoM+ Parallel improves visual grounding across both SAM 2 and MaskFormer segmentation methods when using Gemini 1.5 Pro. However, when using Gemini Flash 2.0 model, the benefit of semantic descriptions appears only in suboptimal segmentation settings (SAM 2 prompt based segmentation).
Metric: Referring Expression Comprehension (REC) Accuracy - bounding box IoU > 50 %
[maskformer_rec_accuracy.pdf](https://github.com/user-attachments/files/19961352/maskformer_rec_accuracy.pdf)
RefCOCOg results with Maskformer segmentation.
[sam2_rec_accuracy.pdf](https://github.com/user-attachments/files/19961353/sam2_rec_accuracy.pdf)
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

demo.ipynb
