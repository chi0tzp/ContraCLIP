#!/bin/bash
# =================================== #
#       Experiment configuration      #
# =================================== #

# ======== GAN Type / Corpus ======== #
gan="stylegan2_ffhq1024"
stylegan_space="W+"
stylegan_layer=11
corpus="expressions3"

# ==== Latent Support Sets (LSS) ==== #
num_latent_support_dipoles=64
min_shift_magnitude=0.1
max_shift_magnitude=0.2
lss_beta=0.5

# ==== Corpus Support Sets (CSS) ==== #
linear=false
styleclip_like=false
loss="cossim"
temperature=0.07
css_beta=0.5
id=true
lambda_id=50

# ============ Training ============= #
batch_size=3
max_iter=20000
# =================================== #


# Run training script
id_loss=""
if $id ; then
  id_loss="--id"
fi

linear_text=""
if $linear ; then
  linear_text="--linear"
fi

styleclip=""
if $styleclip_like ; then
  styleclip="--styleclip"
fi

python train.py --gan=${gan} \
                --truncation=0.7 \
                --stylegan-space=${stylegan_space} \
                --stylegan-layer=${stylegan_layer} \
                --corpus=${corpus} \
                --num-latent-support-dipoles=${num_latent_support_dipoles} \
                --lss-beta=${lss_beta} \
                --loss=${loss} \
                ${id_loss} \
                --lambda-id=${lambda_id} \
                --temperature=${temperature} \
                --css-beta=${css_beta} \
                ${linear_text} \
                ${styleclip} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100
