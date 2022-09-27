#!/bin/bash
# ==================================================================================================================== #
#                                             [ Experiment configuration ]                                             #
# ==================================================================================================================== #

# === GAN Type / Corpus ============================================================================================== #
gan="stylegan2_ffhq1024"         # Choose GAN type from lib/config.py:GENFORCE_MODELS
stylegan_space="W+"              # Choose StyleGAN latent space: Z, W, W+, or S
stylegan_layer=12                # In the case of W+ space, choose up to which layer to use for learning latent paths
corpus="expressions"             # Choose corpus of semantic dipoles from lib/config.py:SEMANTIC_DIPOLES_CORPORA
vl_sim="proposed-no-warping"
no_proj=false

# ====[ Vision-Language Sphere Warper (VLSW) ========================================================================= #
include_cls_in_mean=true
id=true                        # Impose ID preservation using ArcFace
lambda_id=5e3                  # ID preservation loss weighting parameter
gamma_0=1e0
temperature=0.01                # Contrastive loss temperature

# ====[ Latent Support Sets (LSS) ]==================================================================================== #
num_latent_support_dipoles=1   # Set number of support dipoles per support set in the GAN's latent space
min_shift_magnitude=0.05        # set minimum latent shift magnitude
max_shift_magnitude=0.1        # set maximum latent shift magnitude

# === Training ======================================================================================================= #
batch_size=3                   # Set training batch size (cannot be larger than the size of the given corpus)
max_iter=5000                  # Set maximum number of training iterations
lr=1e-1                        # set learning rate for learning the latent support sets LSS (with Adam optimizer)
# ==================================================================================================================== #


include_cls_in_mean_=""
if $include_cls_in_mean ; then
  include_cls_in_mean_="--include-cls-in-mean"
fi

id_=""
if $id ; then
  id_="--id"
fi

no_proj_=""
if $no_proj ; then
  no_proj_="--no-proj"
fi

# ======= Run training script ======= #
python train.py --gan=${gan} \
                --truncation=0.7 \
                --stylegan-space=${stylegan_space} \
                --stylegan-layer=${stylegan_layer} \
                --corpus=${corpus} \
                --vl-sim=${vl_sim} \
                ${include_cls_in_mean_} \
                --gamma-0=${gamma_0} \
                --num-latent-support-dipoles=${num_latent_support_dipoles} \
                --temperature=${temperature} \
                ${id_} \
                ${no_proj_} \
                --lambda-id=${lambda_id} \
                --lr=${lr} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100
