#!/bin/bash

# ===== Configuration =====
pool="stylegan2_ffhq1024-8"
# -------------------------
eps=0.15
shift_leap=5
batch_size=5
# =========================

# Define shift steps
declare -a SHIFT_STEPS=(80)

# Define experiment directories list
declare -a EXPERIMENTS=(
                        "experiments/complete/ablation_temperature/ContraCLIP_stylegan2_ffhq1024-W+6-K2-D32-eps0.1_0.2-beta-lss_0.5-contrastive_0.01-beta-css_0.25+10000.0xID-iter_10000-dev"
                        "experiments/complete/ablation_temperature/ContraCLIP_stylegan2_ffhq1024-W+6-K2-D32-eps0.1_0.2-beta-lss_0.5-contrastive_0.1-beta-css_0.25+10000.0xID-iter_10000-dev"
                        "experiments/complete/ablation_temperature/ContraCLIP_stylegan2_ffhq1024-W+6-K2-D32-eps0.1_0.2-beta-lss_0.5-contrastive_0.25-beta-css_0.25+10000.0xID-iter_10000-dev"
                        "experiments/complete/ablation_temperature/ContraCLIP_stylegan2_ffhq1024-W+6-K2-D32-eps0.1_0.2-beta-lss_0.5-contrastive_0.5-beta-css_0.25+10000.0xID-iter_10000-dev"
                        "experiments/complete/ablation_temperature/ContraCLIP_stylegan2_ffhq1024-W+6-K2-D32-eps0.1_0.2-beta-lss_0.5-contrastive_1.0-beta-css_0.25+10000.0xID-iter_10000-dev"
                        "experiments/complete/ablation_temperature/ContraCLIP_stylegan2_ffhq1024-W+6-K2-D32-eps0.1_0.2-beta-lss_0.5-contrastive_10.0-beta-css_0.25+10000.0xID-iter_10000-dev"
                        )

for shift_s in "${SHIFT_STEPS[@]}"
do
  for exp in "${EXPERIMENTS[@]}"
  do
    python traverse_latent_space.py -v \
                                    --exp="${exp}" \
                                    --pool=${pool} \
                                    --eps=${eps} \
                                    --shift-steps="${shift_s}" \
                                    --shift-leap=${shift_leap} \
                                    --batch-size=${batch_size} \
                                    --img-size=512 \
                                    --gif \
                                    --gif-height=256 \
                                    --strip \
                                    --strip-height=256
  done
done
