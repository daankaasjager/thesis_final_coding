#!/bin/bash

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
source /scratch/s3905845/venvs/thesis/bin/activate
cd /scratch/s3905845/thesis_final_coding

echo "Running property prediction inference for all experiments..."

EXPERIMENTS=(
  "model_size_tiny"
  "model_size_small"
  "ape_70"
  "ape_80"
  "ape_110"
  "prepend_1"
  "prepend_3"
  "prepend_8"
  "prepend_all"
  "embedding_1"
  "embedding_3"
  "embedding_8"
  "embedding_all"
  "cfg_02"
)

# Define guidance scales for cfg_02 only
CFG_02_GUIDANCE_SCALES=(0.3 1.0 4.0)

# Loop through experiments
for EXP_NAME in "${EXPERIMENTS[@]}"
do
  if [[ "$EXP_NAME" == "cfg_02" ]]; then
    # Run cfg_02 with multiple guidance scales
    for GS in "${CFG_02_GUIDANCE_SCALES[@]}"
    do
      echo "Running inference for $EXP_NAME with guidance_scale $GS"

      # Set sampled_data path based on guidance scale (matches your example)
      SAMPLED_DATA="/scratch/s3905845/thesis_final_coding/data/kraken/sampled_data/${GS}_cfg/hist_generated_samples.json"

      # Run inference
      python main.py \
        experiment.name="$EXP_NAME" \
        conditioning.guidance_scale="$GS" \
        property_prediction.inference.hist_sampled_selfies_file="$SAMPLED_DATA" \
        mode="predict_properties" \
        property_prediction.inference.hist=True
    done
  else
    echo "Running inference for $EXP_NAME"

    # Standard inference run for other experiments
    python main.py \
      experiment.name="$EXP_NAME" \
      mode="predict_properties" \
      property_prediction.inference.hist=True
  fi
done

echo "✅ All property predictions completed successfully."
