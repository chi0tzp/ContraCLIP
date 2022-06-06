#!/bin/bash
# =================================== #
#       Experiment configuration      #
# =================================== #

# ======== GAN Type / Corpus ======== #
gan="pggan_celebahq1024"
corpus="attributes"

# ==== Latent Support Sets (LSS) ==== #
num_latent_support_dipoles=128
min_shift_magnitude=0.1
max_shift_magnitude=0.2

# ==== Corpus Support Sets (CSS) ==== #
linear=false
styleclip_like=false
loss="contrastive"
temperature=0.5
beta=0.75

# ============ Training ============= #
batch_size=9
max_iter=120000
# =================================== #


# Run training script
linear_text=""
if $linear ; then
  linear_text="--linear"
fi

styleclip=""
if $styleclip_like ; then
  styleclip="--styleclip"
fi

python train.py --gan=${gan} \
                --corpus=${corpus} \
                --num-latent-support-dipoles=${num_latent_support_dipoles} \
                --loss=${loss} \
                --temperature=${temperature} \
                --beta=${beta} \
                ${linear_text} \
                ${styleclip} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100
