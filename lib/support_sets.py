import sys
import torch
from torch import nn
import numpy as np


class SupportSets(nn.Module):
    def __init__(self, prompt_features=None, num_support_sets=None, num_support_dipoles=None, support_vectors_dim=None,
                 gamma=None, beta=0.5):
        """SupportSets class constructor.

        Args:
            prompt_features (torch.Tensor) : CLIP text feature statistics of prompts from the given corpus
            num_support_sets (int)         : number of support sets (each one defining a warping function)
            num_support_dipoles (int)      : number of support dipoles per support set (per warping function)
            support_vectors_dim (int)      : dimensionality of support vectors (latent space dimensionality, z_dim)
            gamma (float)                  : RBF gamma parameter (by default set to the inverse of the latent space
                                             dimensionality)
        """
        super(SupportSets, self).__init__()
        self.prompt_features = prompt_features

        ################################################################################################################
        ##                                                                                                            ##
        ##                                        [ Corpus Support Sets (CSS) ]                                       ##
        ##                                                                                                            ##
        ################################################################################################################
        if self.prompt_features is not None:
            # Initialization
            self.num_support_sets = self.prompt_features.shape[0]
            self.num_support_dipoles = 1
            self.support_vectors_dim = self.prompt_features.shape[2]

            ############################################################################################################
            ##                                      [ SUPPORT_SETS: (K, N, d) ]                                       ##
            ############################################################################################################
            self.SUPPORT_SETS = nn.Parameter(data=torch.ones(self.num_support_sets,
                                                             2 * self.num_support_dipoles * self.support_vectors_dim),
                                             requires_grad=False)
            self.SUPPORT_SETS.data = self.prompt_features.reshape(self.prompt_features.shape[0],
                                                                  self.prompt_features.shape[1] *
                                                                  self.prompt_features.shape[2]).clone()

            ############################################################################################################
            ##                                          [ ALPHAS: (K, N) ]                                            ##
            ############################################################################################################
            # Define alphas as pairs of [-1, 1] for each dipole
            self.ALPHAS = torch.zeros(self.num_support_sets, 2 * self.num_support_dipoles)
            for k in range(self.num_support_sets):
                a = []
                for _ in range(self.num_support_dipoles):
                    a.extend([1, -1])
                self.ALPHAS[k] = torch.Tensor(a)

            ############################################################################################################
            ##                                          [ GAMMAS: (K, N) ]                                            ##
            ############################################################################################################
            # Define RBF loggammas
            self.LOGGAMMA = nn.Parameter(data=torch.ones(self.num_support_sets, 1), requires_grad=False)
            LOGGAMMA = torch.zeros(self.num_support_sets, 1)
            self.beta = beta
            for k in range(self.num_support_sets):
                g = -np.log(self.beta) / (self.prompt_features[k, 1] - self.prompt_features[k, 0]).norm() ** 2
                LOGGAMMA[k] = torch.log(torch.Tensor([g]))
            self.LOGGAMMA.data = LOGGAMMA.clone()

        ################################################################################################################
        ##                                                                                                            ##
        ##                                       [ Latent Support Sets (LSS) ]                                        ##
        ##                                                                                                            ##
        ################################################################################################################
        else:
            # Initialization
            if num_support_sets is None:
                raise ValueError("Number of latent support sets not defined.")
            else:
                self.num_support_sets = num_support_sets
            if num_support_dipoles is None:
                raise ValueError("Number of latent support dipoles not defined.")
            else:
                self.num_support_dipoles = num_support_dipoles
            if support_vectors_dim is None:
                raise ValueError("Latent support vector dimensionality not defined.")
            else:
                self.support_vectors_dim = support_vectors_dim
            if gamma is None:
                self.gamma = 1.0 / self.support_vectors_dim
            else:
                self.gamma = gamma
            self.loggamma = torch.log(torch.scalar_tensor(self.gamma))

            ############################################################################################################
            ##                                      [ SUPPORT_SETS: (K, N, d) ]                                       ##
            ############################################################################################################
            # TODO
            # Choose r_min and r_max based on the expected latent norm -- i.e., the expected norm of a latent code drawn
            # from the latent space (Z or W) for the given truncation parameter
            self.expected_latent_norm = 9.0128

            # === Z-space ===
            # self.r_min = 0.8 * 22.6186
            # self.r_max = 0.9 * 22.6186
            # === W-space (truncation=1.0) ===
            # self.r_min = 0.8 * 11.0305
            # self.r_max = 0.9 * 11.0305
            # === W-space (truncation=0.7) ===
            # self.r_min = 0.8 * 9.0128
            # self.r_max = 0.9 * 9.0128

            self.r_min = 0.8 * self.expected_latent_norm
            self.r_max = 0.9 * self.expected_latent_norm
            self.radii = torch.arange(self.r_min, self.r_max, (self.r_max - self.r_min) / self.num_support_sets)
            self.SUPPORT_SETS = nn.Parameter(data=torch.ones(self.num_support_sets,
                                                             2 * self.num_support_dipoles * self.support_vectors_dim))
            SUPPORT_SETS = torch.zeros(self.num_support_sets, 2 * self.num_support_dipoles, self.support_vectors_dim)
            for k in range(self.num_support_sets):
                SV_set = []
                for i in range(self.num_support_dipoles):
                    SV = torch.randn(1, self.support_vectors_dim)
                    SV_set.extend([SV, -SV])
                SV_set = torch.cat(SV_set)
                SV_set = self.radii[k] * SV_set / torch.norm(SV_set, dim=1, keepdim=True)
                SUPPORT_SETS[k, :] = SV_set

            # Reshape support sets tensor into a matrix and initialize support sets matrix
            self.SUPPORT_SETS.data = SUPPORT_SETS.reshape(self.num_support_sets,
                                                          2 * self.num_support_dipoles * self.support_vectors_dim).clone()

            ############################################################################################################
            ##                                          [ ALPHAS: (K, N) ]                                            ##
            ############################################################################################################
            # Define alphas as pairs of [-1, 1] for each dipole
            self.ALPHAS = torch.zeros(self.num_support_sets, 2 * self.num_support_dipoles)
            for k in range(self.num_support_sets):
                a = []
                for _ in range(self.num_support_dipoles):
                    a.extend([1, -1])
                self.ALPHAS[k] = torch.Tensor(a)

            ############################################################################################################
            ##                                          [ GAMMAS: (K, N) ]                                            ##
            ############################################################################################################
            # Define RBF loggammas
            self.LOGGAMMA = nn.Parameter(data=self.loggamma * torch.ones(self.num_support_sets, 1))

    def forward(self, support_sets_mask, z):
        # Get RBF support sets batch
        support_sets_batch = torch.matmul(support_sets_mask, self.SUPPORT_SETS)
        support_sets_batch = support_sets_batch.reshape(-1, 2 * self.num_support_dipoles, self.support_vectors_dim)

        # Get batch of RBF alpha parameters
        alphas_batch = torch.matmul(support_sets_mask, self.ALPHAS).unsqueeze(dim=2)

        # Get batch of RBF gamma/log(gamma) parameters
        gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA).unsqueeze(dim=2))

        # Calculate grad of f at z
        D = z.unsqueeze(dim=1).repeat(1, 2 * self.num_support_dipoles, 1) - support_sets_batch

        grad_f = -2 * (alphas_batch * gammas_batch *
                       torch.exp(-gammas_batch * (torch.norm(D, dim=2) ** 2).unsqueeze(dim=2)) * D).sum(dim=1)

        return grad_f / torch.norm(grad_f, dim=1, keepdim=True)
