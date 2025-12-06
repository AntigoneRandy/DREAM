"""
Unified configuration management for DREAM project
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, fields


@dataclass
class DREAMArguments:
    """Unified DREAM configuration arguments"""

    # Model configuration
    t2i_model_type: str = "SD1.5"  # Text-to-image model type
    unet_weight: str = ""  # Optional UNet checkpoint path overriding the base model
    category: str = "sexual"  # NSFW themes (sexual or violence)
    filter_type: Optional[str] = None  # External safety filter type
    num_images_per_prompt: int = 3  # Number of images to generate per prompt

    # LLM model configuration
    llm_model_id: str = "google/gemma-2-27b-it"  # Red-teaming LLM model ID

    # Training configuration
    max_steps: int = 300  # Total number of training steps
    save_steps: int = 300  # Number of steps between checkpoint saves
    eval_steps: int = 25  # Number of steps between evaluations during training
    eval_strategy: str = "steps"  # The evaluation strategy to adopt during training
    save_strategy: str = "steps"  # The checkpoint save strategy to adopt during training
    num_train_epochs: int = 60  # Total number of training epochs
    learning_rate: float = 1e-6  # The initial learning rate for AdamW optimizer
    train_batch_size: int = 32  # Batch size for training
    eval_batch_size: int = 64  # Batch size for evaluation during training

    # Optimization parameters
    zo_eps: float = 1e-3  # Zeroth-order perturbation magnitude
    alpha: float = 0.3  # Temperature scaling parameter for prompt sampling
    gamma: float = 1.2  # Weight for historical gradients in gradient estimation
    temperature: float = 0.8  # Minimum sampling temperature during prompt generation

    # Loss weights
    entropy_weight: float = 0.001  # Weight for prompt entropy regularizer
    similarity_weight: float = 0.5  # Base similarity penalty weight
    # Whether to use adaptive similarity penalty weight
    adaptive_similarity: bool = True
    toxicity_weight: float = 1.0  # Weight for BLIP2-based toxicity metric

    # Generation parameters
    sampling: bool = True  # Whether to use sampling (vs greedy decode)
    top_k: Optional[int] = 200  # Top-k sampling parameter
    top_p: Optional[float] = None  # Top-p sampling parameter
    max_new_tokens: int = 30  # Max number of new tokens to generate

    # Experiment settings
    seed: int = 42  # Global RNG seed
    save_model: bool = True  # Whether to save the model

    def update_from_config(self, config) -> 'DREAMArguments':
        """
        Batch update arguments from config object

        Args:
            config: DREAMConfig object

        Returns:
            Self for method chaining
        """
        # Model configuration
        if hasattr(config, 'model'):
            self.t2i_model_type = config.model.t2i_model_type
            self.unet_weight = config.model.unet_weight or ""

        # Filter configuration
        if hasattr(config, 'filter'):
            self.filter_type = config.filter.filter_type

        # Training configuration
        if hasattr(config, 'training'):
            # Use dataclass fields to avoid hardcoding field names
            training_fields = [f.name for f in fields(
                DREAMArguments.TrainingConfig)]
            for field_name in training_fields:
                if hasattr(config.training, field_name):
                    setattr(self, field_name, getattr(
                        config.training, field_name))

        # Experiment configuration
        if hasattr(config, 'experiment'):
            self.category = config.experiment.category

        return self

    # Legacy config classes for backward compatibility
    @dataclass
    class ModelConfig:
        """Model configuration"""
        t2i_model_type: str = "SD1.5"
        model_id: str = "SD1.5"
        unet_weight: Optional[str] = None
        device: str = "cuda"
        dtype: str = "float16"

    @dataclass
    class FilterConfig:
        """Filter configuration"""
        filter_type: Optional[str] = None

    @dataclass
    class TrainingConfig:
        # Training configuration
        max_steps: int = 300
        save_steps: int = 300
        eval_steps: int = 25
        eval_strategy: str = "steps"
        save_strategy: str = "steps"
        num_train_epochs: int = 60
        learning_rate: float = 1e-6
        train_batch_size: int = 32
        eval_batch_size: int = 64
        gamma: float = 1.2
        alpha: float = 0.3
        top_k: Optional[int] = 200
        similarity_weight: float = 0.5
        adaptive_similarity: bool = True
        toxicity_weight: float = 1.0
        entropy_weight: float = 0.001
        temperature: float = 0.8
        seed: int = 42
        zo_eps: float = 1e-3
        max_new_tokens: int = 30


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str
    category: str
    output_dir: str = ""


@dataclass
class DREAMConfig:
    """Main configuration class"""
    experiment: ExperimentConfig
    model: DREAMArguments.ModelConfig = field(
        default_factory=DREAMArguments.ModelConfig)
    filter: DREAMArguments.FilterConfig = field(
        default_factory=DREAMArguments.FilterConfig)
    training: DREAMArguments.TrainingConfig = field(
        default_factory=DREAMArguments.TrainingConfig)


class ConfigManager:
    """Unified configuration manager for DREAM project"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir

    def load_config(self, config_path: str) -> DREAMConfig:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config file

        Returns:
            DREAMConfig: Loaded configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        return self._parse_config(config_data)

    def _parse_config(self, config_data: Dict[str, Any]) -> DREAMConfig:
        """Parse configuration dictionary into DREAMConfig object"""
        # Parse experiment config
        exp_data = config_data.get('experiment', {})
        experiment = ExperimentConfig(
            name=exp_data.get('name', 'default'),
            category=exp_data.get('category', 'sexual'),
            output_dir=exp_data.get('output_dir', '')
        )

        # Parse model config
        model_data = config_data.get('model', {})
        model = DREAMArguments.ModelConfig(
            t2i_model_type=model_data.get('t2i_model_type', 'SD1.5'),
            model_id=model_data.get('model_id', 'SD1.5'),
            unet_weight=model_data.get('unet_weight'),
            device=model_data.get('device', 'cuda'),
            dtype=model_data.get('dtype', 'float16')
        )

        # Parse filter config
        filter_data = config_data.get('filter', {})
        filter_config = DREAMArguments.FilterConfig(
            filter_type=filter_data.get('filter_type')
        )

        # Parse training config with proper type conversion
        training_data = config_data.get('training', {})

        # Get default values from TrainingConfig
        default_training = DREAMArguments.TrainingConfig()

        # Build training kwargs with proper type conversion
        training_kwargs = {}
        for field_info in fields(DREAMArguments.TrainingConfig):
            field_name = field_info.name
            field_type = field_info.type
            default_value = getattr(default_training, field_name)

            # Get value from config data with default
            config_value = training_data.get(field_name, default_value)

            # Handle None values from YAML
            if config_value is None or config_value == 'None':
                training_kwargs[field_name] = None
            # Convert to proper type
            elif field_type == int:
                training_kwargs[field_name] = int(config_value)
            elif field_type == float:
                training_kwargs[field_name] = float(config_value)
            else:
                training_kwargs[field_name] = config_value

        training = DREAMArguments.TrainingConfig(**training_kwargs)

        return DREAMConfig(
            experiment=experiment,
            model=model,
            filter=filter_config,
            training=training
        )

    def get_config_path(self, category: str, t2i_model_type: str, filter_type: Optional[str] = None) -> str:
        """
        Get configuration file path based on category, model type, and filter type

        Args:
            category: Content category (sexual, violence)
            t2i_model_type: Model type (SD1.5, esd, uce, etc.)
            filter_type: Filter type (sc, text, image, keyword-gibberish, or None)

        Returns:
            str: Path to configuration file
        """
        if filter_type == "sc":
            config_path = f"{self.config_dir}/filters/{category}/safety_checker.yaml"
        elif filter_type == "keyword_gibberish":
            config_path = f"{self.config_dir}/filters/{category}/keyword_gibberish_filter.yaml"
        elif filter_type in ["text", "image"]:
            config_path = f"{self.config_dir}/filters/{category}/nsfw_{filter_type}_detector.yaml"
        else:
            config_path = f"{self.config_dir}/t2i_models/{category}/{t2i_model_type}.yaml"

        return config_path
