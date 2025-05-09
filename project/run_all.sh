#!/bin/bash

echo "Running data preparation..."
python scripts/prepare_data.py
if [ $? -ne 0 ]; then
    echo "Data preparation failed"
    exit 1
fi

echo "Running model training..."
python scripts/train_model.py
if [ $? -ne 0 ]; then
    echo "Model training failed"
    exit 1
fi

echo "Running pipeline parameter optimization..."
python scripts/optimize_pipeline.py
if [ $? -ne 0 ]; then
    echo "Pipeline parameter optimization failed"
    exit 1
fi

echo "Running inference..."
python scripts/run_inference.py
if [ $? -ne 0 ]; then
    echo "Inference failed"
    exit 1
fi

echo "All scripts completed successfully"