#!/bin/bash

# ===== Configuration =====
pool="stylegan2_ffhq1024-8"
# -------------------------
eps=0.15
shift_leap=5
batch_size=10
# =========================

# Define shift steps
declare -a SHIFT_STEPS=(100)

# Define experiment directories list
declare -a EXPERIMENTS=(
                        "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W+-K3-D128-eps0.1_0.2-nonlinear_beta-0.75-contrastive_1.0-10000-complex"
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
                                    --gif-size=512 \
                                    --strip \
                                    --strip-size=256
  done
done
