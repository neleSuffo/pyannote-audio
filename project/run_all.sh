#!/bin/bash

# Check if a model argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 sseriouss"
    echo "Example: $0 pyannet"
    exit 1
fi

MODEL_NAME=$1
echo "Running all scripts for model: $MODEL_NAME"

echo "Running data preparation..."
python scripts/prepare_data.py
if [ $? -ne 0 ]; then
    echo "Data preparation failed"
    exit 1
fi

echo "Running model training for $MODEL_NAME..."
python scripts/train_model.py --model "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo "Model training for $MODEL_NAME failed"
    exit 1
fi

echo "Running pipeline parameter optimization for $MODEL_NAME..."
python scripts/optimize_pipeline.py --model "$MODEL_NAME"
if [ $? -ne 0 ]; then
    echo "Pipeline parameter optimization for $MODEL_NAME failed"
    exit 1
fi

# echo "Running inference for $MODEL_NAME..."
# python scripts/run_inference.py --model "$MODEL_NAME"
# if [ $? -ne 0 ]; then
#     echo "Inference for $MODEL_NAME failed"
#     exit 1
# fi

echo "All scripts completed successfully for model: $MODEL_NAME"