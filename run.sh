#!/bin/bash
# Allow command line config path argument
if [ $# -eq 0 ]; then
    # Default config if no argument provided
    config_path="configs/t2i_models/sexual/sd15.yaml"
elif [ $# -eq 1 ]; then
    # Use provided config path
    config_path="$1"
else
    echo "Usage: $0 [config_path]"
    exit 1
fi

# Check if config file exists
if [ ! -f "$config_path" ]; then
    echo "Error: Config file '$config_path' not found!"
    exit 1
fi


# Extract values from config using sed 
t2i_model_type=$(sed -n 's/.*t2i_model_type: *"\(.*\)"/\1/p' "$config_path")
category=$(sed -n 's/.*category: *"\(.*\)"/\1/p' "$config_path")
alpha=$(sed -n 's/.*alpha: *\([0-9.]*\)/\1/p' "$config_path")
base_dir=$(sed -n 's/.*output_dir: *"\(.*\)"/\1/p' "$config_path")
filter_type=$(sed -n 's/.*filter_type: *"\(.*\)"/\1/p' "$config_path")
unet_weight=$(sed -n 's/.*unet_weight: *"\(.*\)"/\1/p' "$config_path")

timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="${base_dir}/${timestamp}"
log_dir="${output_dir}/logs"
image_dir="${output_dir}/eval/images"
results_dir="${output_dir}/eval/results"
prompt_file_path="${output_dir}/eval/sample.csv"

# Create output directory
mkdir -p "${log_dir}"


exec > >(tee -a "${log_dir}/stdout.log")
exec 2> >(tee -a "${log_dir}/err.log" >&2)

echo "Config: $config_path"
echo "Model Type: $t2i_model_type"
echo "Category: $category"
echo "Unet Weight: $unet_weight"
echo "Output Directory: $output_dir"

# Training
echo "Starting training..."
python src/main.py \
    --load_from_config \
    --config_path "${config_path}" \
    --llm_model_id "google/gemma-2-27b-it" \
    --output_dir "${output_dir}/training"

if [ $? -ne 0 ]; then
    echo "Training failed"
    exit 1
fi

# Sampling
echo "Starting sampling..."
python src/sample.py \
    --category "${category}" \
    --alpha "${alpha}" \
    --model_path "${output_dir}/training/checkpoints/best_model" \
    --sample_batch_size 32 \
    --sample_num_batches 32 \
    --output_dir "${output_dir}/eval"

if [ $? -ne 0 ]; then
    echo "Prompt generation failed"
    exit 1
fi


# Image generation
echo "Starting image generation..."
python src/generate.py \
    --prompt_file_path "${prompt_file_path}" \
    --image_dir "${image_dir}" \
    --unet_weight "${unet_weight}" \
    --t2i_model_type "${t2i_model_type}" \
    --filter_type "${filter_type}" \
    --num_inference_steps 50

if [ $? -eq 0 ]; then
    echo "Image generation completed successfully"
else
    echo "Image generation failed"
    exit 1
fi

# Evaluation
echo "Evaluating generated images..."
python src/eval.py \
    --image_dir "${image_dir}" \
    --results_dir "${results_dir}" \
    --category "${category}" \
    --prompt_file_path "${prompt_file_path}"

if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully"
else
    echo "Evaluation failed"
    exit 1
fi

echo "All processes completed successfully!"
