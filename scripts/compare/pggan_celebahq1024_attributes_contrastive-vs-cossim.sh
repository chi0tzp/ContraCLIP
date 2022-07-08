#!/bin/bash

# ==================================================================================================================== #
declare -a EXPERIMENTAL_RESULTS=(
          "experiments/complete/ContraCLIP_pggan_celebahq1024-Z-K9-D64-lss_beta_0.5-eps0.1_0.2-nonlinear_css_beta_0.5-contrastive_0.07-20000-attributes/results/pggan_celebahq1024-8/120_0.15_18.0/"
          "experiments/complete/ContraCLIP_pggan_celebahq1024-Z-K9-D64-lss_beta_0.5-eps0.1_0.2-nonlinear_css_beta_0.5-cossim-20000-attributes/results/pggan_celebahq1024-8/120_0.15_18.0/"
          )
# ==================================================================================================================== #

e=""
for exp in "${EXPERIMENTAL_RESULTS[@]}"
do
  e+="${exp} "
done

filename=$(basename "$0")
comparison_name="${filename%.*}"
python compare_experiments.py -n "${comparison_name}" -e $e
