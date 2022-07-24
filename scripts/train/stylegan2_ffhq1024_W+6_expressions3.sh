#!/bin/bash
# =================================== #
#       Experiment configuration      #
# =================================== #

# ======== GAN Type / Corpus ======== #
gan="stylegan2_ffhq1024"
stylegan_space="W+"
stylegan_layer=6
corpus="expression3"

# ==== Latent Support Sets (LSS) ==== #
num_latent_support_dipoles=32
min_shift_magnitude=0.1
max_shift_magnitude=0.2

# ==== Corpus Support Sets (CSS) ==== #
loss="cossim"  # "cossim" or "contrastive"
temperature=0.07
learn_css_gammas=false
id=true
lambda_id=1e3

# ============ Training ============= #
batch_size=3
max_iter=15000
# =================================== #


learn_css_gammas_=""
if $learn_css_gammas ; then
  learn_css_gammas_="--learn-css-gammas"
fi

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
                --loss=${loss} \
                ${learn_css_gammas_} \
                --lambda-id=${lambda_id} \
                ${id_} \
                --temperature=${temperature} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100
