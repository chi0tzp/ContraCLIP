#!/bin/bash

# ===== Configuration =====
pool="stylegan2_ffhq1024-8"
# -------------------------
eps=0.15
shift_leap=4
batch_size=10
# =========================

# Define shift steps
declare -a SHIFT_STEPS=(100)

# Define experiment directories list
declare -a EXPERIMENTS=(
                        "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W+-K9-D64-lss_beta_0.5-eps0.1_0.2-nonlinear_css_beta_0.5-contrastive_0.07-20000-attributes"
                        "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W+-K3-D64-lss_beta_0.5-eps0.1_0.2-nonlinear_css_beta_0.5-contrastive_0.07-20000-complex"
                        "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W+-K3-D64-lss_beta_0.5-eps0.1_0.2-nonlinear_css_beta_0.5-contrastive_0.07-20000-expressions3"
                        "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W+-K21-D64-lss_beta_0.5-eps0.1_0.2-nonlinear_css_beta_0.5-contrastive_0.07-20000-expressions"
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
