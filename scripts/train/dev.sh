#!/bin/bash
# =================================== #
#       Experiment configuration      #
# =================================== #

# ======== GAN Type / Corpus ======== #
gan="stylegan2_ffhq1024"
stylegan_space="W+"
stylegan_layer=7
corpus="expressions3"

# ==== Latent Support Sets (LSS) ==== #
num_latent_support_dipoles=32
min_shift_magnitude=0.1
max_shift_magnitude=0.2
lr=1e-3

# ==== Corpus Support Sets (CSS) ==== #
loss="cossim"
temperature=0.07
id=false
lambda_id=10000

# ============ Training ============= #
batch_size=1
max_iter=10000
# =================================== #

# Run training script
id_=""
if $id ; then
  id_="--id"
fi

# ======= Run training script ======= #
python train.py --gan=${gan} \
                --truncation=0.7 \
                --stylegan-space=${stylegan_space} \
                --stylegan-layer=${stylegan_layer} \
                --corpus=${corpus} \
                --num-latent-support-dipoles=${num_latent_support_dipoles} \
                --lr=${lr} \
                --loss=${loss} \
                --lambda-id=${lambda_id} \
                ${id_} \
                --temperature=${temperature} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100
