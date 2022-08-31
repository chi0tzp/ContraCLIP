#!/bin/bash
# ==================================================================================================================== #
#                                             [ Experiment configuration ]                                             #
# ==================================================================================================================== #

# === GAN Type / Corpus ============================================================================================== #
gan="pggan_celebahq1024"       # Choose GAN type from lib/config.py:GENFORCE_MODELS
corpus="attributes-wo-id"      # Choose corpus of semantic dipoles from lib/config.py:SEMANTIC_DIPOLES_CORPORA
vl_paths="non-geodesic"        # Choose type of VL paths ("non-geodesic" or "geodesic")

# ==== Corpus Support Sets (CSS) ===================================================================================== #
id=true                        # Impose ID preservation using ArcFace
lambda_id=1e3                  # ID preservation loss weighting parameter
gamma=1e-6                     # Initialise the gamma parameters of the RBFs in the Vision-Language space
learn_gammas=false             # Optimise CSS RBFs' gammas

# ==== Latent Support Sets (LSS) ===================================================================================== #
num_latent_support_dipoles=32  # Set number of support dipoles per support set in the GAN's latent space
min_shift_magnitude=0.1        # set minimum latent shift magnitude
max_shift_magnitude=0.2        # set maximum latent shift magnitude

# === Training ======================================================================================================= #
batch_size=2                   # Set training batch size (cannot be larger than the size of the given corpus)
max_iter=30000                 # Set maximum number of training iterations
lr=1e-3                        # set learning rate for learning the latent support sets LSS (with Adam optimizer)
# ==================================================================================================================== #


learn_gammas_=""
if $learn_gammas ; then
  learn_gammas_="--learn-gammas"
fi

id_=""
if $id ; then
  id_="--id"
fi

# ======= Run training script ======= #
python train.py --gan=${gan} \
                --truncation=0.7 \
                --corpus=${corpus} \
                --vl-paths=${vl_paths} \
                --gamma=${gamma} \
                --num-latent-support-dipoles=${num_latent_support_dipoles} \
                ${learn_gammas_} \
                ${id_} \
                --lambda-id=${lambda_id} \
                --lr=${lr} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100
