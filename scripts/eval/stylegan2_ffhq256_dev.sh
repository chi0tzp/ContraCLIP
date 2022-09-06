#!/bin/bash

# ===== Configuration =====
pool="stylegan2_ffhq256-8"
# -------------------------
eps=0.15
shift_leap=10
batch_size=10
# =========================

# Define shift steps
# declare -a SHIFT_STEPS=(10 20 30 50)
declare -a SHIFT_STEPS=(70)

# Define experiment directories list
declare -a EXPERIMENTS=(
                        "experiments/wip/ContraCLIP-geodesic-stylegan2_ffhq256-W+8-K2-D128-eps0.1_0.2+1000.0xID-tau_0.07-iter_5000-dev"
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
                                    --gif \
                                    --gif-height=256 \
                                    --strip \
                                    --strip-height=512
  done
done
