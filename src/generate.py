import argparse
import os
import torch
import pandas as pd
from PIL import Image

# Import unified utilities
from utils.t2i_model_loader import load_t2i_model, set_seed
from utils.nsfw_filter import NSFWFilterManager


def generate_images(args):
    """Generate images using the loaded pipeline"""
    # Load T2I model using unified loader

    pipe = load_t2i_model(
        t2i_model_type=args.t2i_model_type,
        unet_weight=args.unet_weight,
        filter_type=getattr(args, 'filter_type', None)
    )

    # Initialize unified filter manager
    filter_type = getattr(args, 'filter_type', None)
    filter_manager = NSFWFilterManager(filter_type, device="cuda") if filter_type else None

    # Load prompts from CSV file
    prompts = pd.read_csv(args.prompt_file_path)['prompt'].tolist()
    os.makedirs(args.image_dir, exist_ok=True)

    print(f"Generating {len(prompts) * args.num_images_per_prompt} images...")
    print(f"Output directory: {args.image_dir}")

    # Process in batches for better memory management
    batch_size = 64
    total_prompts = len(prompts)

    for batch_start in range(0, total_prompts, batch_size):
        # Get current batch of prompts
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_index = batch_start // batch_size

        text_check_results = [False] * len(batch_prompts)  
        image_check_results = [False] * (len(batch_prompts))  
        print(f"Processing batch {batch_index}, prompts {batch_start} to {batch_start + len(batch_prompts) - 1}")


        if filter_manager:
            text_check_results = filter_manager.filter_text(batch_prompts)

        for j in range(args.num_images_per_prompt):
            torch.cuda.empty_cache()

            # Generate images for the current batch
            images = pipe(
                batch_prompts,
                height=512,
                width=512,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            ).images

            # Apply filters if available
            if filter_manager:
                image_check_results = filter_manager.filter_images(images)

            check_ans = [
                text_flag or image_flag 
                for text_flag, image_flag in zip(text_check_results, image_check_results)
            ]

            # Replace filtered images with black images
            for i in range(len(check_ans)):
                if check_ans[i]:
                    images[i] = Image.new('RGB', (512, 512), (0, 0, 0))

            # Save images with proper naming
            for i in range(len(images)):
                prompt_idx = batch_start + i
                file_name = f'{prompt_idx}-{j}.png'
                file_path = os.path.join(args.image_dir, file_name)
                images[i].save(file_path)

        print(f"Batch {batch_index} completed")

    torch.cuda.empty_cache()
    print(f"Generated {len(prompts) * args.num_images_per_prompt} images to {args.image_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using fine-tuned Stable Diffusion")
    parser.add_argument("--prompt_file_path", required=True, help="Path to CSV file containing prompts")
    parser.add_argument("--image_dir", required=True, help="Directory to save generated images")
    parser.add_argument("--t2i_model_type", default="SD1.5", help="Base model path")
    parser.add_argument("--unet_weight", default=None, help="Path to fine-tuned UNet weights")
    parser.add_argument("--num_images_per_prompt", type=int, default=3, help="Number of images to generate per prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--filter_type", default=None, help="Filter type to use")
    parser.add_argument("--category", default="sexual", help="Content category for filtering")

    args = parser.parse_args()

    # Print configuration
    print("=" * 50)
    print("Generation Configuration:")
    print(f"  Prompt file: {args.prompt_file_path}")
    print(f"  Output directory: {args.image_dir}")
    print(f"  T2I model type: {args.t2i_model_type}")
    print(f"  UNet weights: {args.unet_weight or 'Using pretrained'}")
    print(f"  Filter: {args.filter_type}")
    print(f"  Category: {args.category}")
    print("=" * 50)

    set_seed(args.seed)
    generate_images(args)


if __name__ == "__main__":
    main()