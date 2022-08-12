#!/bin/bash

# ===== Configuration =====
pool="stylegan2_ffhq1024-8"
# -------------------------
eps=0.3
shift_leap=10
batch_size=5
# =========================

# Define shift steps
declare -a SHIFT_STEPS=(50)

# Define experiment directories list
declare -a EXPERIMENTS=(
                        "experiments/complete/ablation_beta_css/ContraCLIP_stylegan2_ffhq1024-non-geodesic-W+6-K6-D32-eps0.2_0.4-beta-lss_0.5-beta-css_0.05-iter_20000-expressions6"
                        "experiments/complete/ablation_beta_css/ContraCLIP_stylegan2_ffhq1024-non-geodesic-W+6-K6-D32-eps0.2_0.4-beta-lss_0.5-beta-css_0.15-iter_20000-expressions6"
                        "experiments/complete/ablation_beta_css/ContraCLIP_stylegan2_ffhq1024-non-geodesic-W+6-K6-D32-eps0.2_0.4-beta-lss_0.5-beta-css_0.25-iter_20000-expressions6"
                        "experiments/complete/ablation_beta_css/ContraCLIP_stylegan2_ffhq1024-non-geodesic-W+6-K6-D32-eps0.2_0.4-beta-lss_0.5-beta-css_0.5-iter_20000-expressions6"
                        "experiments/complete/ablation_beta_css/ContraCLIP_stylegan2_ffhq1024-non-geodesic-W+6-K6-D32-eps0.2_0.4-beta-lss_0.5-beta-css_0.75-iter_20000-expressions6"
                        "experiments/complete/ablation_beta_css/ContraCLIP_stylegan2_ffhq1024-non-geodesic-W+6-K6-D32-eps0.2_0.4-beta-lss_0.5-beta-css_0.95-iter_20000-expressions6"
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
