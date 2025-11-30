"""
Unified NSFW filtering utilities for DREAM project
"""

import numpy as np
import re
import pandas as pd
from typing import List, Union, Optional
from PIL import Image
from transformers import pipeline
from contextlib import redirect_stdout
import io
from skimage import transform
from dataclasses import dataclass


@dataclass
class FilterConfig:
    """Configuration for content filters"""
    filter_type: Optional[str] = None
    device: str = "cuda"


class BaseFilter:
    """Base class for content filters"""

    def __init__(self, config: FilterConfig):
        self.config = config
        self.device = config.device
        self.filter_type = config.filter_type

    def filter_content(self, inputs: Union[List[str], List[Image.Image]]) -> List[bool]:
        """
        Apply filter to content

        Args:
            inputs: List of text strings or images

        Returns:
            List[bool]: True if content should be filtered/blocked
        """
        raise NotImplementedError("Subclasses must implement filter_content")


class SCFilter(BaseFilter):
    """
    Safety Checker Filter using Stable Diffusion Safety Checker
    """

    def __init__(self, config: FilterConfig):
        super().__init__(config)
        self._init_safety_checker()

    def _init_safety_checker(self):
        """Initialize safety checker model"""
        from transformers import AutoFeatureExtractor
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id).to(self.device)
        self.safety_checker.eval()

    def check_safety(self, x_image: np.ndarray) -> List[bool]:
        """
        Check safety of images using Stable Diffusion safety checker

        Args:
            x_image: Batch of images as numpy array [batch, H, W, C]

        Returns:
            List[bool]: True if NSFW content detected
        """
        # Convert numpy to PIL
        pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in x_image]

        # Process through safety checker
        safety_checker_input = self.safety_feature_extractor(pil_images, return_tensors="pt")
        safety_checker_input = {k: v.to(self.device) for k, v in safety_checker_input.items()}

        x_checked_image, has_nsfw_concept = self.safety_checker(
            clip_input=safety_checker_input['pixel_values'],
            images=x_image
        )

        return has_nsfw_concept

    def filter_content(self, inputs: Union[List[str], List[Image.Image]]) -> List[bool]:
        """
        Apply safety checker filter to images

        Args:
            inputs: List of PIL Images

        Returns:
            List[bool]: True if content should be filtered/blocked
        """
        if isinstance(inputs[0], str):
            # SC filter only works on images, return False for text
            return [False] * len(inputs)

        # Convert PIL images to numpy array
        np_images = []
        for img in inputs:
            if isinstance(img, Image.Image):
                np_img = np.array(img).astype(np.float32) / 255
                np_images.append(np_img)
            else:
                # Assume it's already a numpy array
                np_images.append(img.astype(np.float32) / 255)

        np_images = np.stack(np_images)

        # Check safety
        has_nsfw = self.check_safety(np_images)

        return has_nsfw


class TextFilter(BaseFilter):
    """
    Text-based NSFW content filter using DistilBERT
    """

    def __init__(self, config: FilterConfig):
        super().__init__(config)
        self._init_text_classifier()

    def _init_text_classifier(self):
        """Initialize text classification model"""
        self.text_classifier = pipeline(
            "text-classification",
            model="eliasalbouzidi/distilbert-nsfw-text-classifier",
            device=self.device if self.device != "cpu" else -1
        )

    def filter_content(self, inputs: Union[List[str], List[Image.Image]]) -> List[bool]:
        """
        Apply text filter to text content

        Args:
            inputs: List of text strings

        Returns:
            List[bool]: True if content should be filtered/blocked
        """
        if isinstance(inputs[0], Image.Image):
            # Text filter only works on text, return False for images
            return [False] * len(inputs)

        predictions = self.text_classifier(inputs)
        return [pred['label'] == 'nsfw' for pred in predictions]


class ImageFilter(BaseFilter):
    """
    Image-based NSFW content filter using TensorFlow model
    """

    def __init__(self, config: FilterConfig):
        super().__init__(config)
        self._init_image_classifier()

    def _init_image_classifier(self):
        """Initialize image classification model"""
        import tensorflow as tf

        # Configure TensorFlow memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        from keras.models import load_model
        model_path = "files/checkpoints/nsfw_detector/image_classifier.h5"
        self.nsfw_image_model = load_model(model_path, compile=False)

    def filter_content(self, inputs: Union[List[str], List[Image.Image]]) -> List[bool]:
        """
        Apply image filter to image content

        Args:
            inputs: List of PIL Images

        Returns:
            List[bool]: True if content should be filtered/blocked
        """
        if isinstance(inputs[0], str):
            # Image filter only works on images, return False for text
            return [False] * len(inputs)

        # Preprocess images
        processed = []
        for img in inputs:
            if isinstance(img, Image.Image):
                np_img = np.array(img).astype(np.float32) / 255
            else:
                np_img = img.astype(np.float32) / 255
            resized_img = transform.resize(np_img, (224, 224, 3))
            processed.append(resized_img)

        # Batch prediction
        batch = np.array(processed)
        with redirect_stdout(io.StringIO()):  # Suppress model output
            preds = self.nsfw_image_model.predict(batch)

        return [np.argmax(p) != 0 for p in preds]


class KeywordGibberishFilter(BaseFilter):
    """
    Hybrid filter combining keyword-based and gibberish detection
    """

    def __init__(self, config: FilterConfig):
        super().__init__(config)
        self._init_filters()

    def _init_filters(self):
        """Initialize keyword and gibberish detection models"""
        # Initialize keyword patterns
        self.keywords = {
            "sexual", "nude", "sex", "18+", "naked", "nsfw",
            "porn", "dick", "vagina", "explicit", "uncensored",
            "fuck", "nipples", "areola", "blood", "weapons",
            "gun", "kill", "murder", "gore", "mutilation",
            "shoot", "dead", "violent", "slaughter", "bomb", "knife",
            "hips", "seductive", "underwear", "bra", "ass",
        }

        # Build regex patterns
        self.patterns = []
        for keyword in self.keywords:
            escaped_keyword = re.escape(keyword)
            pattern = re.compile(
                r'(?<!\w)' + escaped_keyword + r'(?!\w)',
                flags=re.IGNORECASE
            )
            self.patterns.append(pattern)

        # Initialize gibberish classifier
        try:
            self.gibberish_classifier = pipeline(
                "text-classification",
                model="madhurjindal/autonlp-Gibberish-Detector-492513457",
                device=self.device if self.device != "cpu" else -1
            )
        except Exception as e:
            print(f"Warning: Could not load gibberish classifier: {e}")
            self.gibberish_classifier = None

    def _keyword_filter(self, texts: List[str]) -> List[bool]:
        """Apply keyword-based filtering"""
        results = []
        for text in texts:
            text = str(text) if not pd.isna(text) else ""
            has_keyword = any(pattern.search(text) for pattern in self.patterns)
            results.append(has_keyword)
        return results

    def _gibberish_filter(self, texts: List[str]) -> List[bool]:
        """Apply gibberish detection"""
        if self.gibberish_classifier is None:
            return [False] * len(texts)

        classifier_results = self.gibberish_classifier(texts)
        is_gibberish = [
            result['label'] in ['noise', 'word salad', 'mild gibberish']
            for result in classifier_results
        ]
        return is_gibberish

    def filter_content(self, inputs: Union[List[str], List[Image.Image]]) -> List[bool]:
        """
        Apply hybrid keyword-gibberish filter to text content

        Args:
            inputs: List of text strings

        Returns:
            List[bool]: True if content should be filtered/blocked
        """
        if isinstance(inputs[0], Image.Image):
            # This filter only works on text, return False for images
            return [False] * len(inputs)

        # Apply both filters and combine results
        keyword_results = self._keyword_filter(inputs)
        gibberish_results = self._gibberish_filter(inputs)

        # Content is filtered if it matches keywords OR is gibberish
        combined_results = [kw or gb for kw, gb in zip(keyword_results, gibberish_results)]

        return combined_results


class NSFWFilterManager:
    """
    Manager for multiple NSFW filters
    """

    def __init__(self, filter_types: Union[str, List[str]], device: str = "cuda"):
        """
        Initialize filter manager

        Args:
            filter_types: Single filter type string or list of filter types to initialize
            device: Device to run models on
        """
        # Convert string to list if needed
        if isinstance(filter_types, str):
            # Split by comma and strip whitespace
            filter_types = [ft.strip() for ft in filter_types.split(',') if ft.strip()]
        elif filter_types is None:
            filter_types = []

        self.filters = {}
        self.filter_types = filter_types

        for filter_type in filter_types:
            config = FilterConfig(
                filter_type=filter_type,
                device=device,
            )

            if filter_type == "sc":
                self.filters[filter_type] = SCFilter(config)
            elif filter_type == "text":
                self.filters[filter_type] = TextFilter(config)
            elif filter_type == "image":
                self.filters[filter_type] = ImageFilter(config)
            elif filter_type == "keyword-gibberish":
                self.filters[filter_type] = KeywordGibberishFilter(config)
            else:
                print(f"Warning: Unknown filter type '{filter_type}'")

    def filter_content(self, inputs: Union[List[str], List[Image.Image]]) -> List[bool]:
        """
        Apply all configured filters to content

        Args:
            inputs: List of text strings or images

        Returns:
            List[bool]: True if content should be filtered/blocked by any filter
        """
        if not self.filters:
            return [False] * len(inputs)

        # Initialize results as False (not filtered)
        combined_results = [False] * len(inputs)

        # Apply each filter and combine results with OR logic
        for filter_name, filter_obj in self.filters.items():
            try:
                filter_results = filter_obj.filter_content(inputs)
                # Content is filtered if ANY filter says it should be
                combined_results = [combined or filtered for combined, filtered in zip(combined_results, filter_results)]
            except Exception as e:
                print(f"Error applying filter '{filter_name}': {e}")
                continue

        return combined_results

    def filter_text(self, texts: List[str]) -> List[bool]:
        """Convenience method for text filtering"""
        return self.filter_content(texts)

    def filter_images(self, images: List[Image.Image]) -> List[bool]:
        """Convenience method for image filtering"""
        return self.filter_content(images)