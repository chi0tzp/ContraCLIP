import torch
from torch import nn


class CorpusSupportSets(nn.Module):
    def __init__(self, prompt_features, dipole_betas, learn_gammas=False):
        """CorpusSupportSets class constructor.

        Args:
            prompt_features (torch.Tensor) : CLIP text feature statistics of prompts from the given corpus
            dipole_betas (list)            : TODO: +++
        """
        super(CorpusSupportSets, self).__init__()
        self.prompt_features = prompt_features
        self.dipole_betas = dipole_betas
        self.learn_gammas = learn_gammas

        ################################################################################################################
        ##                                                                                                            ##
        ##                                        [ Corpus Support Sets (CSS) ]                                       ##
        ##                                                                                                            ##
        ################################################################################################################
        # Initialization
        self.num_support_sets = self.prompt_features.shape[0]
        self.support_vectors_dim = self.prompt_features.shape[2]

        ############################################################################################################
        ##                                      [ SUPPORT_SETS: (K, 2, d) ]                                       ##
        ############################################################################################################
        self.SUPPORT_SETS = nn.Parameter(data=torch.ones(self.num_support_sets, 2 * self.support_vectors_dim),
                                         requires_grad=False)
        self.SUPPORT_SETS.data = self.prompt_features.reshape(self.prompt_features.shape[0],
                                                              self.prompt_features.shape[1] *
                                                              self.prompt_features.shape[2]).clone()

        ############################################################################################################
        ##                                          [ ALPHAS: (K, 2) ]                                            ##
        ############################################################################################################
        # Define alphas as pairs of [-1, 1] for each dipole
        self.ALPHAS = nn.Parameter(data=torch.zeros(self.num_support_sets, 2), requires_grad=False)
        for k in range(self.num_support_sets):
            self.ALPHAS.data[k] = torch.Tensor([1, -1])

        ############################################################################################################
        ##                                          [ GAMMAS: (K, 2) ]                                            ##
        ############################################################################################################
        # Define RBF loggammas
        self.LOGGAMMA = nn.Parameter(data=torch.ones(self.num_support_sets, 2), requires_grad=self.learn_gammas)
        for k in range(self.num_support_sets):
            betas = self.dipole_betas[k]
            gammas = -torch.log(torch.Tensor(betas)) / \
                (self.prompt_features[k, 1] - self.prompt_features[k, 0]).norm() ** 2
            self.LOGGAMMA.data[k] = torch.log(gammas)

    def forward(self, support_sets_mask, z):
        # Get RBF support sets batch
        support_sets_batch = torch.matmul(support_sets_mask, self.SUPPORT_SETS)
        support_sets_batch = support_sets_batch.reshape(-1, 2, self.support_vectors_dim)

        # Get batch of RBF alpha parameters
        alphas_batch = torch.matmul(support_sets_mask, self.ALPHAS).unsqueeze(dim=2)

        # Get batch of RBF gamma/log(gamma) parameters
        gammas_batch = torch.exp(torch.matmul(support_sets_mask, self.LOGGAMMA).unsqueeze(dim=2))

        # Calculate grad of f at z
        D = z.unsqueeze(dim=1).repeat(1, 2, 1) - support_sets_batch

        grad_f = -2 * (alphas_batch * gammas_batch *
                       torch.exp(-gammas_batch * (torch.norm(D, dim=2) ** 2).unsqueeze(dim=2)) * D).sum(dim=1)

        return grad_f / torch.norm(grad_f, dim=1, keepdim=True)
