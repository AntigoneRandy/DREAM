import os
import argparse
import pandas as pd
from tqdm import tqdm

# Import unified utilities
from utils.llm_generator import load_llm_model, LLMGenerator, set_seed


def parse_sample_args():
    """Parse arguments for sample.py"""
    parser = argparse.ArgumentParser(description="Generate prompts using trained LLM model")

    # Model parameters
    parser.add_argument("--category", type=str, default="sexual",
                       help="Content category (sexual, violence)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to fine-tuned model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for generated prompts")

    # Generation parameters
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="Alpha parameter for sampling")

    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature for sampling")
    
    parser.add_argument("--top_k", type=int, default=200,
                       help="Top-k for generation")

    parser.add_argument("--sample_batch_size", type=int, default=32,
                       help="Batch size for generation")
                       
    parser.add_argument("--sample_num_batches", type=int, default=32,
                       help="Number of batches to generate")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")


    return parser.parse_args()


def main():
    args = parse_sample_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Load data
    results_df = pd.read_csv(f"files/data/{args.category}.csv")
    pool = [(row["prompt"]) for _, row in results_df.iterrows()]

    # Load LLM model and tokenizer
    llm_model, tokenizer = load_llm_model(
        llm_model_id=args.model_path,
    )

    # Initialize LLM generator
    mask_path = "files/checkpoints/mask/mask.npy"
    llm_generator = LLMGenerator(
        model=llm_model,
        tokenizer=tokenizer,
        args=args,
        mask_path=mask_path
    )

    set_seed(args.seed)

    # Generate prompts
    all_prompts = []

    for batch_index in tqdm(range(args.sample_num_batches), desc="Processing batches"):
        # Generate prompts using LLMGenerator
        instructions = [llm_generator.generate_instruction(pool) for _ in range(args.sample_batch_size)]
        prompts, _ = llm_generator.generate_prompts(instructions)
        all_prompts.extend(prompts)

    # Save generated prompts to CSV file
    df = pd.DataFrame(all_prompts, columns=["prompt"])
    save_file = os.path.join(args.output_dir, "sample.csv")

    df.to_csv(save_file, index=False)
    print(f"Saved {len(all_prompts)} prompts to {save_file}")


if __name__ == "__main__":
    main()
