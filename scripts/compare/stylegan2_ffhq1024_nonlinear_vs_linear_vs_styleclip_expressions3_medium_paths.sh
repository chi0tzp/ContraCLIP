#!/bin/bash

# ==================================================================================================================== #
declare -a EXPERIMENTAL_RESULTS=(
          "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W-K3-D128-eps0.1_0.2-nonlinear_beta-0.75-contrastive_0.5-expressions3/results/stylegan2_ffhq1024-8/96_0.15_14.4/"
          "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W-K3-D128-eps0.1_0.2-linear-contrastive_0.5-expressions3/results/stylegan2_ffhq1024-8/96_0.15_14.4/"
          "experiments/complete/ContraCLIP_stylegan2_ffhq1024-W-K3-D128-eps0.1_0.2-styleclip-contrastive_0.5-expressions3/results/stylegan2_ffhq1024-8/96_0.15_14.4/"
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
