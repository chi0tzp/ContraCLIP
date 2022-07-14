#!/bin/bash

# ==================================================================================================================== #
config="140_0.15_21.0"
declare -a EXPERIMENTAL_RESULTS=(
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+2-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+3-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+4-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+5-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+6-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+7-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+8-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+9-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+10-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+11-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+12-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
    "experiments/complete/ablation_stylegan2_W+_space/ContraCLIP_stylegan2_ffhq1024-W+13-K3-D32-lss_beta_0.5-eps0.1_0.2-css_beta_0.5-cossim-15000-expressions3/results/stylegan2_ffhq1024-8/${config}"
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
