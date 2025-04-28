# SegmentationPipelineFactory.py

from .AbstractSegmentationPipeline import AbstractSegmentationPipeline
from .MaskFormerPipeline import MaskFormerPipeline
from .SAM2Pipeline import SAM2Pipeline
from typing import Dict, Any, Optional

class SegmentationPipelineFactory:
    """
    Factory class for creating segmentation pipelines based on model type.
    """
    
    @staticmethod
    def create_pipeline(
        model_type: str,
        model: Any,
        processor: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AbstractSegmentationPipeline:
        """
        Creates the appropriate pipeline based on model type.
        
        Args:
            model_type: String identifier ("maskformer" or "sam2")
            model: The model object
            processor: The processor object (needed for MaskFormer)
            config: Optional configuration parameters
            
        Returns:
            An instance of the appropriate pipeline
        """
        if config is None:
            config = {}
            
        model_type = model_type.lower()
        
        if model_type == "maskformer":
            if processor is None:
                raise ValueError("MaskFormer pipeline requires a processor")
            return MaskFormerPipeline(
                model=model,
                processor=processor,
                **config
            )
            
        elif model_type == "sam2":
            return SAM2Pipeline(
                predictor=model,  # For SAM2, the model is the predictor
                **config
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
