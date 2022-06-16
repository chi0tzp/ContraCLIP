#!/bin/bash

# ===== Configuration =====
pool="pggan_celebahq1024-8"
# -------------------------
eps=0.15
shift_leap=3
batch_size=10
# =========================

# Define shift steps
declare -a SHIFT_STEPS=(50)


# Define experiment directories list
declare -a EXPERIMENTS=(
                        "experiments/complete/ContraCLIP_pggan_celebahq1024-Z-K8-D128-eps0.1_0.2-nonlinear_beta-0.75-contrastive_1.0-10000-attributes"
                        "experiments/complete/ContraCLIP_pggan_celebahq1024-Z-K11-D128-eps0.1_0.2-nonlinear_beta-0.75-contrastive_1.0-10000-expressions"
                        )

for shift_s in "${SHIFT_STEPS[@]}"
do
  for exp in "${EXPERIMENTS[@]}"
  do
    python traverse_latent_space.py -v --gif \
                                    --exp="${exp}" \
                                    --pool=${pool} \
                                    --eps=${eps} \
                                    --shift-steps="${shift_s}" \
                                    --shift-leap=${shift_leap} \
                                    --batch-size=${batch_size} \
                                    --img-size=512 \
                                    --gif \
                                    --gif-size=256 \
                                    --strip \
                                    --strip-size=256
  done
done
