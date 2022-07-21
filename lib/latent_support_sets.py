import torch
from torch import nn


class LatentSupportSets(nn.Module):
    def __init__(self, num_support_sets=None, num_support_dipoles=None, support_vectors_dim=None, beta=0.35,
                 jung_radius=None):
        """LatentSupportSets class constructor.

        Args:
            num_support_sets (int)    : number of support sets (each one defining a warping function)
            num_support_dipoles (int) : number of support dipoles per support set (per warping function)
            support_vectors_dim (int) : dimensionality of support vectors (latent space dimensionality, z_dim)
            beta (float)              : set beta parameter for initializing latent space RBFs' gamma parameters
                                        (0.25 < lss_beta < 1.0)
            jung_radius (float)       : radius of the minimum enclosing ball of a set of a set of 10K latent codes
        """
        super(LatentSupportSets, self).__init__()

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
        if jung_radius is None:
            raise ValueError("Jung radius not given.")
        else:
            self.jung_radius = jung_radius
        self.beta = beta

        ############################################################################################################
        ##                                      [ SUPPORT_SETS: (K, N, d) ]                                       ##
        ############################################################################################################
        # Choose r_min and r_max based on the Jung radius
        self.r_min = 0.90 * self.jung_radius
        self.r_max = 0.95 * self.jung_radius
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
        self.SUPPORT_SETS.data = SUPPORT_SETS.reshape(
            self.num_support_sets, 2 * self.num_support_dipoles * self.support_vectors_dim).clone()

        ############################################################################################################
        ##                                          [ ALPHAS: (K, N) ]                                            ##
        ############################################################################################################
        # Define alphas as pairs of [-1, 1] for each dipole
        self.ALPHAS = nn.Parameter(data=torch.zeros(self.num_support_sets, 2 * self.num_support_dipoles),
                                   requires_grad=False)
        for k in range(self.num_support_sets):
            a = []
            for _ in range(self.num_support_dipoles):
                a.extend([1, -1])
            self.ALPHAS.data[k] = torch.Tensor(a)

        ############################################################################################################
        ##                                          [ GAMMAS: (K, 2) ]                                            ##
        ############################################################################################################
        # Define RBF loggammas
        self.LOGGAMMA = nn.Parameter(data=torch.ones(self.num_support_sets, 2 * self.num_support_dipoles))
        for k in range(self.num_support_sets):
            gammas = -torch.log(torch.Tensor([self.beta, self.beta])) / ((2 * self.radii[k]) ** 2)
            loggammas = torch.log(gammas)
            lg = []
            for _ in range(self.num_support_dipoles):
                lg.extend(loggammas)
            self.LOGGAMMA.data[k] = torch.Tensor(lg)

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
