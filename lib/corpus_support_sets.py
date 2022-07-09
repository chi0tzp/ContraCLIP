import torch
from torch import nn
import numpy as np


class CorpusSupportSets(nn.Module):
    def __init__(self, prompt_features=None, beta=0.5):
        """CorpusSupportSets class constructor.

        Args:
            prompt_features (torch.Tensor) : CLIP text feature statistics of prompts from the given corpus
            beta (float)                   : set beta parameter for fixing CLIP space RBFs' gamma parameters
                                             (0.25 <= css_beta < 1.0)

        """
        super(CorpusSupportSets, self).__init__()
        self.prompt_features = prompt_features

        ################################################################################################################
        ##                                                                                                            ##
        ##                                        [ Corpus Support Sets (CSS) ]                                       ##
        ##                                                                                                            ##
        ################################################################################################################
        # Initialization
        self.num_support_sets = self.prompt_features.shape[0]
        self.num_support_dipoles = 1
        self.support_vectors_dim = self.prompt_features.shape[2]
        self.beta = beta

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
        for k in range(self.num_support_sets):
            g = -np.log(self.beta) / (self.prompt_features[k, 1] - self.prompt_features[k, 0]).norm() ** 2
            self.LOGGAMMA.data[k] = torch.log(torch.Tensor([g]))

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