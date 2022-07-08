#!/bin/bash

# ==================================================================================================================== #
declare -a EXPERIMENTAL_RESULTS=(
          "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W+-K3-D64-lss_beta_0.5-eps0.1_0.2-nonlinear_css_beta_0.5-contrastive_0.07-20000-expressions3/results/stylegan2_ffhq1024-8/200_0.15_30.0/"
          "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W+-K3-D64-lss_beta_0.5-eps0.1_0.2-nonlinear_css_beta_0.5-cossim-20000-expressions3/results/stylegan2_ffhq1024-8/200_0.15_30.0/"
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
