# SoM+ Package

A Python toolkit that combines the Set-of-Mark prompting with additional feature augmentations for each of the annotations. Segmentation is performed with MaskFormer/SAM2 and then augmented via Gemini 2.0 Flash.

## Prerequisites

- **Python** 3.10  

## Installation

1. **Clone & Install _som_plus_**

   ```bash
   git clone https://github.com/<your-org>/som_plus.git
   cd som_plus
   pip install -e .

2. Clone & Install SAM2

cd ..
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

3. Download SAM2 checkpoint

mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt \
     -O checkpoints/sam2.1_hiera_base_plus.pt


## Demo 

demo.ipynb
