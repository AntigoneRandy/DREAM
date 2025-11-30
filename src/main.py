from typing import List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from transformers import TrainingArguments, HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from trainer import OurTrainer
import pandas as pd


# Import unified utilities
from utils.llm_generator import load_llm_model, set_seed
from utils.t2i_model_loader import T2IModelLoader
from utils.config_manager import ConfigManager, DREAMConfig, DREAMArguments

@dataclass
class OurArguments(DREAMArguments, TrainingArguments):
    # Configuration file settings
    config_path: Optional[str] = None  # path to config file in configs/
    load_from_config: bool = False  # whether to load parameters from config file

    # TrainingArguments overrides


def parse_args():
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Load configuration from file if specified
    if args.load_from_config:
        config_manager = ConfigManager()

        # If config_path is not specified, construct it from category and t2i_model_type
        if not args.config_path:
            args.config_path = config_manager.get_config_path(
                category=args.category,
                model_type=args.t2i_model_type.lower(),
                filter_type=args.filter_type
            )

        print(f"Loading configuration from: {args.config_path}")
        config = config_manager.load_config(args.config_path)

        # Override args with config values
        args = _update_args_from_config(args, config)

    print(args)
    return args


def _update_args_from_config(args: OurArguments, config: DREAMConfig) -> OurArguments:
    """Update args object with values from config using batch update"""
    return args.update_from_config(config)




class TrainingFramework:
    def __init__(self, args, model_path=None, config_path=None):
        """
        Initialize training framework with unified model loading

        Args:
            args: Training arguments
            model_path: Optional custom model path
            config_path: Optional config file path for loading from configs/
        """
        self.args = args
        self.t2i_loader = T2IModelLoader(device="cuda")

        # Initialize config manager if config path is provided
        if config_path:
            self.config_manager = ConfigManager()
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = None
            self.config_manager = None

        # Load LLM model
        llm_model_path = model_path if model_path is not None else args.llm_model_id
        self.llm_model, self.tokenizer = load_llm_model(
            llm_model_id=llm_model_path,
        )

        # Load T2I model
        self._initialize_t2i_model()

    def _initialize_t2i_model(self):
        """Initialize T2I model using unified loader"""
        # Determine model type and filter type
        t2i_model_type = self.args.t2i_model_type
        filter_type = getattr(self.args, 'filter_type', None)
        unet_weight = getattr(self.args, 'unet_weight', None)

        # Override with config values if available
        if self.config:
            t2i_model_type = self.config.model.t2i_model_type
            filter_type = self.config.filter.filter_type
            unet_weight = self.config.model.unet_weight


        self.t2i_model = self.t2i_loader.load_t2i_model(
            t2i_model_type=t2i_model_type,
            unet_weight=unet_weight,
            filter_type=filter_type
        )

    def train(self):

        results_df = pd.read_csv(f"files/data/{self.args.category}.csv")
        pool = [(row["prompt"]) for _, row in results_df.iterrows()]
        trainer = OurTrainer(
            llm_model=self.llm_model,
            t2i_model=self.t2i_model,
            tokenizer=self.tokenizer,
            pool=pool,
            args=self.args,
        )

        if self.args.resume_from_checkpoint is not None:
            trainer.train(resume_from_checkpoint=self.args.resume_from_checkpoint)
        else:
            trainer.train()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Determine config path for training framework
    config_path = args.config_path if args.load_from_config else None

    framework = TrainingFramework(args, config_path=config_path)
    framework.train()


if __name__ == "__main__":
    main()
