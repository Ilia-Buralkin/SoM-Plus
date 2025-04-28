 # som_augmentation package

# Import main classes
from .geminin_augmentation import (
    augment_annotations_parallel,
    augment_annotations_unified
)

# You can choose not to expose utility functions like encode_image_from_pil
# unless they're likely to be used by package users
# from .geminin_augmentation import encode_image_from_pil

# Define what's available when someone uses: from som_augmentation import *
__all__ = [
    'augment_annotations_parallel',
    'augment_annotations_unified'
]
