#!/bin/bash
# =================================== #
#       Experiment configuration      #
# =================================== #

# ======== GAN Type / Corpus ======== #
gan="stylegan2_ffhq1024"
stylegan_space="W+"
stylegan_layer=8
corpus="attributes"

# ==== Latent Support Sets (LSS) ==== #
num_latent_support_dipoles=32
min_shift_magnitude=0.1
max_shift_magnitude=0.2
lss_beta=0.5
lr=1e-3

# ==== Corpus Support Sets (CSS) ==== #
loss="cossim"
temperature=0.07
css_beta=0.5

# ============ Training ============= #
batch_size=7
max_iter=15000
# =================================== #


# ======= Run training script ======= #
python train.py --gan=${gan} \
                --truncation=0.7 \
                --stylegan-space=${stylegan_space} \
                --stylegan-layer=${stylegan_layer} \
                --corpus=${corpus} \
                --num-latent-support-dipoles=${num_latent_support_dipoles} \
                --lss-beta=${lss_beta} \
                --lr=${lr} \
                --loss=${loss} \
                --temperature=${temperature} \
                --css-beta=${css_beta} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100
