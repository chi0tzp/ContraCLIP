#!/bin/bash
# ==================================================================================================================== #
#                                             [ Experiment configuration ]                                             #
# ==================================================================================================================== #

# === GAN Type / Corpus ============================================================================================== #
gan="stylegan2_ffhq1024"       # Choose GAN type from lib/config.py:GENFORCE_MODELS
stylegan_space="W+"            # Choose StyleGAN latent space: Z, W, W+, or S
stylegan_layer=6               # In the case of W+ space, choose up to which layer to use for learning latent paths
corpus="expressions6"          # Choose corpus of semantic dipoles from lib/config.py:SEMANTIC_DIPOLES_CORPORA
vl_paths="non-geodesic"        # Choose type of VL paths ("non-geodesic" or "geodesic")

# ==== Corpus Support Sets (CSS) ===================================================================================== #
learn_css_gammas=true          # Optimise CSS RBFs' gammas
id=true                        # Impose ID preservation using ArcFace
lambda_id=10000                # ID preservation loss weighting parameter
beta_css=0.15                  # Set the beta parameter for initialising the gamma parameters of the RBFs in the
                               # CLIP Vision-Language space

# ==== Latent Support Sets (LSS) ===================================================================================== #
num_latent_support_dipoles=32  # Set number of support dipoles per support set in the GAN's latent space
min_shift_magnitude=0.2        # set minimum latent shift magnitude
max_shift_magnitude=0.4        # set maximum latent shift magnitude
beta_lss=0.5                   # set the beta parameter for initialising the gamma parameters of the RBFs in the
                               # GAN's latent space

# === Training ======================================================================================================= #
batch_size=6                   # Set training batch size (cannot be larger than the size of the given corpus)
max_iter=30000                 # Set maximum number of training iterations
lr=1e-3                        # set learning rate for learning the latent support sets LSS (with Adam optimizer)
# ==================================================================================================================== #


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
                --vl-paths=${vl_paths} \
                --beta-css=${beta_css} \
                --num-latent-support-dipoles=${num_latent_support_dipoles} \
                --beta-lss=${beta_lss} \
                ${learn_css_gammas_} \
                ${id_} \
                --lambda-id=${lambda_id} \
                --lr=${lr} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100
