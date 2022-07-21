#!/bin/bash

# ===== Configuration =====
pool="stylegan2_ffhq1024-8"
# pool="stylegan2_ffhq1024-2"
# -------------------------
eps=0.15
shift_leap=7
batch_size=5
# =========================

# Define shift steps
declare -a SHIFT_STEPS=(70)

# Define experiment directories list
declare -a EXPERIMENTS=(
                        "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W+7-K3-D64-eps0.1_0.2-contrastive_0.07-iter_10000-expressions3"
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
